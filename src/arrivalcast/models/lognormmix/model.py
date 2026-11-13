"""
Intensity-free temporal point process with a log-normal mixture decoder
(Shchur, Biloš & Günnemann, 2020).

Core idea: instead of parameterizing an intensity function lambda(t) and
integrating it (as Poisson/Hawkes/Neural Hawkes all do), directly
parameterize the predictive distribution of the next inter-arrival gap
Delta t_{i+1} | history as a mixture of log-normals, whose parameters are
emitted by an RNN that has consumed Delta t_1 .. Delta t_i. This avoids the
Monte Carlo integration Neural Hawkes needs, at the cost of committing to a
specific distributional family per step.

Like Neural Hawkes, this model's per-step RNN input is free-form, so
calendar features / Time2Vec are legitimate inputs here too (see
models/lognormmix/dataloader.py for where they're concatenated in) — unlike
the classical Poisson/Hawkes baselines.

The survival term for the right-censored tail after the last event is
handled via the mixture's survival function S(t) = 1 - CDF(t), added to the
log-likelihood as log S(tail_hours | last hidden state) rather than via
Monte Carlo integration.
"""
from __future__ import annotations

import math

import torch
from torch import nn

_LOG_SQRT_2PI = 0.5 * math.log(2 * math.pi)
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0


class LogNormMixDecoder(nn.Module):
    """
    Maps an RNN hidden state to the parameters of a K-component log-normal
    mixture over the next inter-arrival gap.
    """

    def __init__(self, hidden_dim: int, n_components: int = 4):
        super().__init__()
        self.n_components = n_components
        self.head = nn.Linear(hidden_dim, n_components * 3)  # weight, mean, log-std per component

    def get_params(self, h: torch.Tensor):
        """
        h : (..., hidden_dim)
        Returns (log_weights, means, log_stds), each (..., n_components),
        log_weights already log-softmax-normalized so they sum to 1 in
        probability space.
        """
        raw = self.head(h)
        log_w_raw, means, log_stds = raw.chunk(3, dim=-1)
        log_weights = torch.log_softmax(log_w_raw, dim=-1)
        # FIX: the old upper bound (5.0, i.e. sigma up to e^5 ~ 148 on the log
        # scale) let a low-weight component spread mass over astronomically
        # long gaps, which made CRPS ~1e117 on every source. sigma <= e^1 ~ 2.7
        # still allows a very wide component (a factor of ~15 per s.d.).
        log_stds = log_stds.clamp(min=LOG_STD_MIN, max=LOG_STD_MAX)
        return log_weights, means, log_stds

    def log_prob(self, h: torch.Tensor, gap_hours: torch.Tensor) -> torch.Tensor:
        """
        Log-density of the mixture at the observed gap, log p(Delta t | h).

        gap_hours : (...,) actual inter-arrival gap in hours (must be > 0).
        """
        log_weights, means, log_stds = self.get_params(h)
        log_gap = torch.log(gap_hours.clamp_min(1e-8)).unsqueeze(-1)  # (..., 1)

        # log-normal log-density in terms of log(gap):
        # log p(x) = -log(x) - log(sigma) - 0.5*log(2pi) - 0.5*((log x - mu)/sigma)^2
        z = (log_gap - means) / log_stds.exp()
        component_log_prob = (
            -log_gap - log_stds - _LOG_SQRT_2PI - 0.5 * z**2
        )  # (..., n_components)
        return torch.logsumexp(log_weights + component_log_prob, dim=-1)

    def log_survival(self, h: torch.Tensor, gap_hours: torch.Tensor) -> torch.Tensor:
        """
        log S(t) = log P(next gap > t | h), for the right-censored tail
        after the last observed event. Computed via the standard normal
        CDF applied to the log-transformed gap, mixed the same way as
        log_prob but with survival (1 - CDF) instead of density.
        """
        log_weights, means, log_stds = self.get_params(h)
        log_gap = torch.log(gap_hours.clamp_min(1e-8)).unsqueeze(-1)
        z = (log_gap - means) / log_stds.exp()
        # Survival of a standard normal at z, via the complementary error
        # function for numerical stability in the tail: Phi(-z) = 0.5*erfc(z/sqrt(2))
        component_survival = 0.5 * torch.erfc(z / math.sqrt(2.0))
        component_survival = component_survival.clamp_min(1e-12)
        mixture_survival = (log_weights.exp() * component_survival).sum(dim=-1)
        return torch.log(mixture_survival.clamp_min(1e-12))

    def mean_gap(self, h: torch.Tensor) -> torch.Tensor:
        """
        Point prediction for the next gap: E[Delta t | h] under the
        mixture. Mean of a log-normal component is exp(mu + sigma^2/2).
        """
        log_weights, means, log_stds = self.get_params(h)
        component_means = torch.exp(means + 0.5 * (log_stds.exp() ** 2))
        return (log_weights.exp() * component_means).sum(dim=-1)


class LogNormMix(nn.Module):
    """
    GRU + LogNormMixDecoder. Input at each step is log(Δt_i) optionally
    concatenated with calendar features / Time2Vec of event i (see
    dataloader.py) — the hidden state after step i is used to predict the
    distribution of Δt_{i+1}.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32, n_components: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = LogNormMixDecoder(hidden_dim, n_components)
        # FIX: standardize the log-gap input (column 0). Raw log-gaps range
        # from about -1 (hourly sources) to about 7.6 (quarterly), so the GRU
        # saw very differently scaled inputs per source. Set from the TRAIN
        # split via set_input_normalization(); defaults are a no-op.
        self.register_buffer("in_mean", torch.zeros(1))
        self.register_buffer("in_std", torch.ones(1))

    def set_input_normalization(self, log_gaps_train) -> None:
        """Standardize the log-gap input with train-split statistics and start
        each mixture component's location at the train mean log-gap, so
        training begins from a sensible predictive distribution instead of
        one centred on a 1-hour gap."""
        lg = torch.as_tensor(log_gaps_train, dtype=torch.float32)
        mean = lg.mean()
        std = lg.std() if lg.numel() > 1 else torch.tensor(1.0)
        self.in_mean.fill_(float(mean))
        self.in_std.fill_(float(max(float(std), 1e-3)))
        K = self.decoder.n_components
        with torch.no_grad():
            bias = self.decoder.head.bias
            # head output layout: [log-weights (K) | means (K) | log-stds (K)]
            bias[K:2 * K] = float(mean) + torch.linspace(-0.5, 0.5, K)
            bias[2 * K:] = 0.0

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        x[..., 0] = (x[..., 0] - self.in_mean) / self.in_std
        return x

    def forward(self, inputs: torch.Tensor):
        """
        inputs : (batch, seq_len, input_dim) — step i encodes Δt_i (and
            optional covariates), used to predict Δt_{i+1}.

        Returns
        -------
        hidden_states : (batch, seq_len, hidden_dim) where hidden_states[:,
            i] is h after consuming inputs[:, i], i.e. the state used to
            predict gap i+1 (and, for i = seq_len-1, the tail survival).
        """
        hidden_states, _ = self.rnn(self._normalize(inputs))
        return hidden_states

    def _state_after_last_gap(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor,
        terminal_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Hidden state after the RNN has consumed the sequence's LAST observed
        gap (Delta t_n).

        hidden_states[:, lengths-1] has only consumed Delta t_{n-1} (inputs
        are gaps 1..n-1) and is the state that PREDICTS Delta t_n -- using it
        for the right-censored tail after Delta t_n (or for "the next gap"
        after the last event) is one step behind. So take that state and run
        one more GRU step on terminal_inputs (the features of Delta t_n).
        For a sequence with no supervised steps (lengths == 0) the state
        before consuming Delta t_n is the zero initial state.

        terminal_inputs : (batch, input_dim)
        """
        batch = terminal_inputs.shape[0]
        if hidden_states.shape[1] == 0:
            prev = hidden_states.new_zeros(batch, self.hidden_dim)
        else:
            gather_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
            prev = hidden_states.gather(1, gather_idx).squeeze(1)
            prev = torch.where((lengths > 0).view(-1, 1), prev, torch.zeros_like(prev))
        _, h_n = self.rnn(self._normalize(terminal_inputs.unsqueeze(1)), prev.unsqueeze(0).contiguous())
        return h_n.squeeze(0)  # (batch, hidden_dim)

    def log_likelihood(
        self,
        inputs: torch.Tensor,
        target_gaps: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        tail_hours: torch.Tensor,
        terminal_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        inputs : (batch, seq_len, input_dim) — encodes Δt_1..Δt_{n-1} (see
            forward()). Offset by one relative to target_gaps: inputs[:, i]
            predicts target_gaps[:, i] = Δt_{i+2} (1-indexed).
        target_gaps : (batch, seq_len) the gap each hidden state must
            predict.
        mask : (batch, seq_len) 1.0 for real (non-padded) predictions.
        lengths : (batch,) true number of predicted gaps per sequence.
        tail_hours : (batch,) right-censored tail length after the last
            observed event.
        terminal_inputs : (batch, input_dim) features of the LAST observed
            gap Δt_n (see dataloader.pad_batch). The tail survival must be
            conditioned on a state that has consumed Δt_n, so one extra
            RNN step is taken here (see _state_after_last_gap).

        Returns
        -------
        (batch,) total sequence log-likelihood = sum of per-gap log
        densities + log survival of the tail.
        """
        hidden_states = self.forward(inputs)
        log_probs = self.decoder.log_prob(hidden_states, target_gaps)  # (batch, seq_len)
        seq_ll = (log_probs * mask).sum(dim=1)

        final_hidden = self._state_after_last_gap(hidden_states, lengths, terminal_inputs)
        log_surv = self.decoder.log_survival(final_hidden, tail_hours)
        return seq_ll + log_surv

    def predict_next_gap(
        self, inputs: torch.Tensor, lengths: torch.Tensor, terminal_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Point forecast for the next gap after each sequence's last
        observed event — the production inference path. Conditions on the
        state after consuming the last gap (not the one that predicted it)."""
        hidden_states = self.forward(inputs)
        final_hidden = self._state_after_last_gap(hidden_states, lengths, terminal_inputs)
        return self.decoder.mean_gap(final_hidden)
