"""
Neural Hawkes Process (Mei & Eisner, 2017), single-type (mark-free) version.

Core idea: a continuous-time LSTM (CTLSTM) maintains a hidden state that
decays exponentially *between* events (rather than jumping discretely like
a normal RNN), so the intensity function lambda(t) is well-defined at every
real-valued t, not just at observed event times. This is what lets the
model capture history-dependent, non-monotonic intensity shapes beyond what
a fixed exponential-kernel Hawkes process can express.

Unlike models/hawkes and models/poisson, this model's per-step input is a
free-form vector — so calendar features (features/calendar_features.py)
and/or a Time2Vec encoding (features/time2vec.py) of the elapsed time can
be concatenated in directly. See models/neural_hawkes/dataloader.py for
where that happens.

Because the neural intensity has no closed-form integral, the compensator
term in the log-likelihood is estimated via Monte Carlo integration using
sample points drawn within each inter-event interval (see dataloader).

This is a compact, single-file implementation of the CTLSTM cell and
intensity head; it omits multi-type embeddings (not needed - each source is
single-type by this project's scoping) and the "future event" attention
variants from later papers, staying close to the original formulation.
"""
from __future__ import annotations

import torch
from torch import nn


class CTLSTMCell(nn.Module):
    """
    One continuous-time LSTM cell (Mei & Eisner, 2017, Eq. 3-7).

    Standard LSTM gates plus a *decay* cell state: instead of a single
    cell state c, the cell tracks (c, c_bar, decay_rate) so that the
    hidden state used for the intensity at any time t after the last event
    can be computed by decaying c towards c_bar at the learned rate.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        gate_input_dim = input_dim + hidden_dim
        # input, forget, output, z (cell candidate), input-bar, forget-bar, decay
        self.linear = nn.Linear(gate_input_dim, hidden_dim * 7)

    def forward(self, x, h_prev, c_prev, c_bar_prev):
        """
        x : (batch, input_dim) event/covariate embedding at this step
        h_prev, c_prev, c_bar_prev : (batch, hidden_dim) state carried over
            from the previous event. h_prev AND c_prev must both already be
            decayed to "just before" this event (decay_cell()/decay_to());
            c_bar is constant between events so needs no decay.

        Returns h, c, c_bar, decay_rate — all (batch, hidden_dim); c/c_bar/
        decay_rate parameterize how the state evolves until the next event.
        """
        combined = torch.cat([x, h_prev], dim=-1)
        (i, f, o, z, i_bar, f_bar, decay_raw) = self.linear(combined).chunk(7, dim=-1)

        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        z = torch.tanh(z)
        i_bar, f_bar = torch.sigmoid(i_bar), torch.sigmoid(f_bar)
        decay_rate = nn.functional.softplus(decay_raw)

        c = f * c_prev + i * z
        c_bar = f_bar * c_bar_prev + i_bar * z
        h = o * torch.tanh(c)  # hidden state immediately after the event
        return h, c, c_bar, o, decay_rate

    @staticmethod
    def decay_cell(c, c_bar, decay_rate, elapsed):
        """Cell state `elapsed` time after the event that produced
        (c, c_bar, decay_rate): c(t) = c_bar + (c - c_bar) * exp(-decay * dt)
        (Mei & Eisner, 2017, Eq. 7)."""
        return c_bar + (c - c_bar) * torch.exp(-decay_rate * elapsed)

    @staticmethod
    def decay_to(c, c_bar, decay_rate, o, elapsed):
        """
        Decay the cell state forward by `elapsed` time and derive the
        hidden state at that point — used both to get h just before the
        *next* event, and to evaluate the intensity at Monte Carlo
        integration sample points between events.

        elapsed : (batch, 1) or (batch, n_samples, 1) time since the event
            that produced (c, c_bar, decay_rate, o).
        """
        c_t = CTLSTMCell.decay_cell(c, c_bar, decay_rate, elapsed)
        h_t = o * torch.tanh(c_t)
        return h_t


class NeuralHawkes(nn.Module):
    """
    Full model: embeds each event's inputs, runs the CTLSTM, and reads the
    (single-type) intensity off the hidden state via a softplus-activated
    linear layer, as in the original paper's Eq. 8.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = CTLSTMCell(input_dim, hidden_dim)
        self.intensity_weight = nn.Linear(hidden_dim, 1, bias=True)
        # Typical gap (hours) used to rescale the Time2Vec input; set from the
        # train split (median gap) before training. See dataloader.build_event_inputs.
        self.register_buffer("gap_scale", torch.ones(1))

    def init_from_train_gaps(self, gaps_hours) -> None:
        """Set gap_scale and start the intensity near the train-split rate
        (softplus(b) = 1/mean gap), so early training is not spent climbing
        from an intensity that is orders of magnitude off."""
        import numpy as np
        g = np.asarray(gaps_hours, dtype=float)
        g = g[g > 0]
        if g.size == 0:
            return
        self.gap_scale.fill_(float(np.median(g)))
        rate = 1.0 / float(np.mean(g))
        with torch.no_grad():
            self.intensity_weight.bias.fill_(float(np.log(np.expm1(rate))) if rate < 20 else rate)

    def intensity_from_hidden(self, h):
        """lambda(t) = softplus(w^T h(t)) — always positive, as required."""
        return nn.functional.softplus(self.intensity_weight(h)).squeeze(-1)

    def forward(self, inputs, gaps, integration_gaps, lengths=None):
        """
        inputs : (batch, seq_len, input_dim) per-event feature vectors
            (e.g. Time2Vec(Δt) concatenated with calendar features — see
            dataloader.py for construction).
        gaps : (batch, seq_len) Δt_i = t_i - t_{i-1}, in hours, aligned so
            that `inputs[:, i]` and `gaps[:, i]` both describe the interval
            ending at event i.
        integration_gaps : (batch, seq_len, n_mc_samples) times, each
            measured *since the previous event*, of Monte Carlo sample
            points drawn within interval i (used only to estimate the
            compensator integral for that interval, not fed back as model
            input).
        lengths : optional (batch,) long tensor of the true (unpadded)
            sequence length per source, required to correctly pick out the
            post-last-real-event state for the tail survival term when
            batching sources of very different lengths together (see
            note in log_likelihood/tail_compensator). If None, every
            sequence in the batch is assumed to have length seq_len (no
            padding) and the state after the loop's last step is used
            directly.

        Returns
        -------
        log_intensities : (batch, seq_len) — see loop comments below.
        compensators : (batch, seq_len) Monte Carlo estimate of
            integral_{t_{i-1}}^{t_i} lambda(s) ds for each interval i.
        final_state_per_seq : (c, c_bar, o, decay_rate), each (batch,
            hidden_dim) — the state immediately AFTER the true last real
            event of each sequence (correctly gathered via `lengths`,
            not just the state after the last padded step).
        all_states : (c, c_bar, o, decay_rate), each (batch, seq_len,
            hidden_dim) — the state immediately after EVERY real step,
            not just the last. Training (log_likelihood) only needs
            final_state_per_seq (for the tail survival term) and ignores
            this; models/neural_hawkes/evaluate.py uses it to build a
            predictive distribution at every test-split position, the
            same way LogNormMix.forward() exposes its full hidden-state
            sequence for the analogous purpose.
        """
        batch, seq_len, _ = inputs.shape
        device = inputs.device
        h = torch.zeros(batch, self.hidden_dim, device=device)
        c = torch.zeros(batch, self.hidden_dim, device=device)
        c_bar = torch.zeros(batch, self.hidden_dim, device=device)
        o = torch.zeros(batch, self.hidden_dim, device=device)
        decay_rate = torch.ones(batch, self.hidden_dim, device=device)

        log_intensities = []
        compensators = []
        # Stashed per-step post-update state, so we can gather the state
        # at each sequence's own true last event afterwards.
        c_hist, c_bar_hist, o_hist, decay_hist = [], [], [], []

        for i in range(seq_len):
            gap_i = gaps[:, i : i + 1]  # (batch, 1)

            # Intensity AT event i: decay the state from event i-1 forward
            # by the full gap, then read off lambda just before the event.
            c_before_event = CTLSTMCell.decay_cell(c, c_bar, decay_rate, gap_i)
            h_before_event = o * torch.tanh(c_before_event)
            lam_at_event = self.intensity_from_hidden(h_before_event)
            log_intensities.append(torch.log(lam_at_event.clamp_min(1e-10)))

            # Monte Carlo compensator for interval i: sample points at
            # integration_gaps[:, i, :] (time since event i-1), decay the
            # SAME pre-event state to each sample time, average intensity,
            # multiply by interval length gap_i.
            samples = integration_gaps[:, i, :]  # (batch, n_mc_samples)
            h_samples = CTLSTMCell.decay_to(
                c.unsqueeze(1), c_bar.unsqueeze(1), decay_rate.unsqueeze(1), o.unsqueeze(1),
                samples.unsqueeze(-1),
            )
            lam_samples = self.intensity_from_hidden(h_samples)  # (batch, n_mc_samples)
            compensators.append(lam_samples.mean(dim=-1) * gap_i.squeeze(-1))

            # Update state using the actual event i's input. BOTH the hidden
            # state and the cell state must be the values decayed to the
            # event time (Mei & Eisner: c(t_i) enters the update, not the
            # cell value c_{i-1} from right after the previous event).
            # Feeding the undecayed c here silently ignores the decay that
            # the intensity itself was just computed with.
            h, c, c_bar, o, decay_rate = self.cell(
                inputs[:, i], h_before_event, c_before_event, c_bar
            )
            c_hist.append(c)
            c_bar_hist.append(c_bar)
            o_hist.append(o)
            decay_hist.append(decay_rate)

        if lengths is None:
            final_state = (c, c_bar, o, decay_rate)
        else:
            gather_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1)  # (batch, 1, 1)
            c_stack = torch.stack(c_hist, dim=1)  # (batch, seq_len, hidden)
            c_bar_stack = torch.stack(c_bar_hist, dim=1)
            o_stack = torch.stack(o_hist, dim=1)
            decay_stack = torch.stack(decay_hist, dim=1)
            idx = gather_idx.expand(-1, 1, self.hidden_dim)
            final_state = tuple(
                s.gather(1, idx).squeeze(1) for s in (c_stack, c_bar_stack, o_stack, decay_stack)
            )

        all_states = (
            torch.stack(c_hist, dim=1), torch.stack(c_bar_hist, dim=1),
            torch.stack(o_hist, dim=1), torch.stack(decay_hist, dim=1),
        )  # each (batch, seq_len, hidden_dim) — the state immediately
        # AFTER each real step i, for evaluators (models/neural_hawkes/
        # evaluate.py) that need a prediction at every position, not just
        # the sequence's final one (unlike training, which only needs
        # final_state for the tail survival term).

        return torch.stack(log_intensities, dim=1), torch.stack(compensators, dim=1), final_state, all_states

    def tail_compensator(self, final_state, tail_integration_samples):
        """
        Monte Carlo estimate of the compensator for the right-censored tail
        interval [t_n, T] after the last real event — required so the
        model is correctly penalized/credited for "no event occurred"
        between the last observed arrival and the end of the observation
        window (see core/windows.py docstring on why T must be the study
        end, not the last event).

        final_state : (c, c_bar, o, decay_rate) returned by forward() for
            the last real event in each sequence (dataloader must select
            the right index per-sequence if sequences are padded to
            different lengths within a batch).
        tail_integration_samples : (batch, n_mc_samples) times since the
            last real event, sampled within [0, tail_hours].

        Returns
        -------
        (batch,) compensator estimate for the tail interval. Multiply the
        mean sampled intensity by tail_hours (the caller/dataloader passes
        pre-scaled samples and this method assumes the samples are drawn
        uniformly over [0, tail_hours], so scaling is applied by the caller
        the same way as the per-interval case in forward()).
        """
        c, c_bar, o, decay_rate = final_state
        c_e, c_bar_e, o_e, decay_e = (t.unsqueeze(1) for t in final_state)
        h_samples = CTLSTMCell.decay_to(
            c_e, c_bar_e, decay_e, o_e, tail_integration_samples.unsqueeze(-1)
        )
        lam_samples = self.intensity_from_hidden(h_samples)
        return lam_samples.mean(dim=-1)  # caller multiplies by tail_hours

    def log_likelihood(
        self, inputs, gaps, integration_gaps, mask, lengths, tail_hours, tail_integration_samples
    ):
        """
        mask : (batch, seq_len) 1.0 for real events, 0.0 for padding —
            needed because sources have very different sequence lengths
            (e.g. SRC_08 has ~17.5k events, SRC_23 has 2) and are batched
            together with padding.
        lengths : (batch,) true sequence length per source — see forward().
        tail_hours : (batch,) length of the right-censored tail per source.
        tail_integration_samples : (batch, n_mc_samples) — see
            tail_compensator().

        log-likelihood = sum_i log lambda(t_i) - integral_0^T lambda(s) ds
        where the integral is split into per-event-interval compensators
        (from forward()) plus the tail compensator (from tail_compensator()).
        """
        log_intensities, compensators, final_state, _ = self.forward(
            inputs, gaps, integration_gaps, lengths=lengths
        )
        per_event_ll = log_intensities - compensators
        seq_ll = (per_event_ll * mask).sum(dim=1)

        tail_mean_lam = self.tail_compensator(final_state, tail_integration_samples)
        seq_ll = seq_ll - tail_mean_lam * tail_hours
        return seq_ll
