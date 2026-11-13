"""
Training loops for the two neural models, shared by notebooks/initial_experiments.ipynb
and the experiment runner (arrivalcast.experiments.runner) so both produce identical models.

What changed relative to the previous version
---------------------------------------------
1. **Chunked (truncated-BPTT) training.** Each source is a single sequence, so
   the old loop took ONE full-batch gradient step per "epoch" and every step
   unrolled the CT-LSTM over every training event in a Python loop (SRC_08:
   ~30 s per step). Now a long training sequence is cut into windows of about
   ``chunk_len`` events; windows are shuffled and trained in mini-batches, so
   one epoch is many cheap gradient steps. Each window is preceded by
   ``burn_in`` events that warm up the recurrent state but contribute no loss,
   which keeps training close to evaluation (evaluation replays the whole
   history, so the state at test time is always "warm").
   A sequence shorter than ``2 * chunk_len`` stays a single window, i.e. the
   old full-batch behaviour.
2. **All hyper-parameters live in ``TrainConfig``** instead of module
   constants, so a run is described by one object that can be logged, hashed
   and varied (tuning, ablations, seeds).
3. **The Neural-Hawkes step cap is gone** (it was a separate 50-step constant).
   Both models use ``max_epochs`` / ``patience`` and early-stop on validation.
4. Optional ``weight_decay``; non-finite losses skip the step instead of
   poisoning the weights.
5. ``info`` now also reports ``best_epoch``, ``n_steps`` and ``hit_cap`` (True
   if training ended because ``max_epochs`` ran out rather than because
   validation stopped improving), so under-trained runs are visible.

Objective note: a sequence that fits in ONE window (fewer than 2 * chunk_len
positions) uses the exact full-sequence objective including the censored tail,
as before. Only when a sequence is split into several windows is the tail term
dropped from training/validation (it is then one term out of hundreds). The
reported ``test_ll`` always includes the tail. ``best_val_*`` values are not
comparable with the first preliminary run for the windowed sources.
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
from torch.optim import Adam

from arrivalcast.features.time2vec import RawGapEncoding, Time2Vec
from arrivalcast.models.lognormmix.dataloader import (
    LogNormMixDataLoader,
    build_step_inputs,
)
from arrivalcast.models.lognormmix.dataloader import (
    pad_batch as lnm_pad_batch,
)
from arrivalcast.models.lognormmix.model import LogNormMix
from arrivalcast.models.neural_hawkes.dataloader import NeuralHawkesDataLoader
from arrivalcast.models.neural_hawkes.dataloader import pad_batch as nhp_pad_batch
from arrivalcast.models.neural_hawkes.model import NeuralHawkes


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    # --- architecture ---
    hidden_dim: int = 32
    t2v_dim: int = 8              # NHP only
    n_components: int = 4         # LogNormMix only
    n_mc_samples: int = 5         # NHP only: Monte-Carlo points per interval
    use_calendar_features: bool = True  # evaluation must use the same flag
    use_time2vec: bool = True     # NHP only. False = feed the raw scaled gap instead
                                  # (ablation; LogNormMix has no Time2Vec, so ignores this)

    # --- optimisation ---
    lr: float = 1e-2
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    max_epochs: int = 200         # passes over the training windows
    patience: int = 20            # epochs without validation improvement
    min_delta: float = 1e-4

    # --- windowing (truncated BPTT) ---
    chunk_len: int = 128          # target events per window (loss part)
    burn_in: int = 32             # warm-up events before each window, no loss
    batch_size: int = 16          # windows per gradient step

    seed: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def config_id(self, ignore: tuple = ("seed",)) -> str:
        """Short stable hash of the settings (seed excluded by default), used to
        name run directories so results from different settings never mix."""
        d = {k: v for k, v in self.to_dict().items() if k not in ignore}
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


DEFAULT_CONFIG = TrainConfig()

# Backwards-compatible aliases (notebooks/initial_experiments.ipynb prints these).
HIDDEN_DIM = DEFAULT_CONFIG.hidden_dim
MAX_EPOCHS = DEFAULT_CONFIG.max_epochs
PATIENCE = DEFAULT_CONFIG.patience
LR = DEFAULT_CONFIG.lr


def make_time_encoder(cfg: TrainConfig) -> torch.nn.Module:
    """The Neural Hawkes gap encoder: Time2Vec, or the raw-gap ablation."""
    return Time2Vec(out_dim=cfg.t2v_dim) if cfg.use_time2vec else RawGapEncoding()


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# Windowing helpers
# --------------------------------------------------------------------------
def make_spans(n: int, chunk_len: int, burn_in: int) -> list[tuple[int, int, int]]:
    """Cut positions 0..n-1 into windows. Returns (a, s, b) triples: the model
    is fed positions [a, b) and the loss is taken on [s, b); [a, s) is burn-in.

    The number of windows is n // chunk_len (at least 1), so every window has
    between chunk_len and 2*chunk_len loss positions and there is no tiny
    trailing remainder. n < 2*chunk_len therefore gives one window.
    """
    if n <= 0:
        return []
    n_chunks = max(1, n // max(chunk_len, 1))
    bounds = np.linspace(0, n, n_chunks + 1).round().astype(int)
    return [(max(int(s) - burn_in, 0), int(s), int(b)) for s, b in zip(bounds[:-1], bounds[1:])]


def _collate_indices(spans):
    """Padded index/mask tensors for a list of (a, s, b) windows."""
    B = len(spans)
    W = max(b - a for a, _, b in spans)
    idx = torch.zeros(B, W, dtype=torch.long)
    valid = torch.zeros(B, W)
    loss_mask = torch.zeros(B, W)
    for i, (a, s, b) in enumerate(spans):
        w = b - a
        idx[i, :w] = torch.arange(a, b)
        valid[i, :w] = 1.0
        loss_mask[i, s - a:w] = 1.0
    return idx, valid, loss_mask


# --------------------------------------------------------------------------
# Generic fitting loop
# --------------------------------------------------------------------------
def _fit(params, modules, train_spans, step_loss_fn, val_fn, cfg: TrainConfig):
    """Minibatch Adam over shuffled windows with validation early stopping.
    Restores the best-validation weights into `modules` before returning."""
    optimizer = Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    best, best_state, best_epoch, bad = np.inf, None, 0, 0
    epochs_run, n_steps, hit_cap = 0, 0, True

    for epoch in range(1, cfg.max_epochs + 1):
        for m in modules:
            m.train()
        order = rng.permutation(len(train_spans))
        train_losses = []
        for i in range(0, len(order), cfg.batch_size):
            batch = [train_spans[j] for j in order[i:i + cfg.batch_size]]
            optimizer.zero_grad()
            loss = step_loss_fn(batch)
            if not torch.isfinite(loss):
                continue  # skip a poisoned step rather than corrupt the weights
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            optimizer.step()
            n_steps += 1
            train_losses.append(loss.item())
        epochs_run = epoch

        for m in modules:
            m.eval()
        with torch.no_grad():
            v = val_fn() if val_fn is not None else (
                float(np.mean(train_losses)) if train_losses else np.inf)
        if v < best - cfg.min_delta:
            best, best_epoch, bad = v, epoch, 0
            best_state = [copy.deepcopy(m.state_dict()) for m in modules]
        else:
            bad += 1
            if bad >= cfg.patience:
                hit_cap = False
                break

    if best_state is not None:
        for m, st in zip(modules, best_state):
            m.load_state_dict(st)
    for m in modules:
        m.eval()
    return dict(best=float(best), best_epoch=best_epoch, epochs_run=epochs_run,
                n_steps=n_steps, hit_cap=hit_cap, n_chunks=len(train_spans))


# --------------------------------------------------------------------------
# Neural Hawkes
# --------------------------------------------------------------------------
def _nhp_ll(model, time2vec, seq, rng, gap_scale, n_mc):
    """Full-sequence log-likelihood INCLUDING the tail term (used for test_ll)."""
    b = nhp_pad_batch([seq], time2vec, n_mc, rng, gap_scale)
    return model.log_likelihood(
        b["inputs"], b["gaps"], b["integration_gaps"], b["mask"], b["lengths"],
        b["tail_hours"], b["tail_integration_samples"],
    ).sum()


def _nhp_window_loss(model, time2vec, gaps_t, cal_t, spans, gap_scale, n_mc, generator=None):
    """Mean negative log-likelihood per loss-event over a batch of windows."""
    idx, valid, loss_mask = _collate_indices(spans)
    g = gaps_t[idx] * valid                                   # (B, W) hours
    feats = time2vec((g / gap_scale).unsqueeze(-1))           # (B, W, t2v_dim)
    if cal_t.shape[1] > 0:
        feats = torch.cat([feats, cal_t[idx]], dim=-1)
    u = torch.rand(g.shape[0], g.shape[1], n_mc, generator=generator)
    integ = u * g.unsqueeze(-1)                               # MC points inside each gap
    log_int, comp, _, _ = model.forward(feats, g, integ)
    ll = ((log_int - comp) * loss_mask).sum()
    return -ll / loss_mask.sum().clamp_min(1.0)


def train_nhp_one_source(df, global_study_end, cfg: TrainConfig | None = None):
    """Returns (test_ll, n_test_events, model, time2vec, info) or None."""
    cfg = cfg or DEFAULT_CONFIG
    _seed(cfg.seed)
    loader = NeuralHawkesDataLoader(
        df["arrival_timestamp"], use_calendar_features=cfg.use_calendar_features,
        n_mc_samples=cfg.n_mc_samples, global_study_end=global_study_end,
    )
    splits = loader.get_splits()
    train_seq, val_seq, test_seq = splits["train"], splits["val"], splits["test"]
    if train_seq.n_events < 5:
        return None

    time2vec = make_time_encoder(cfg)
    n_cal = train_seq.calendar_features.shape[1]
    model = NeuralHawkes(input_dim=time2vec.out_dim + n_cal, hidden_dim=cfg.hidden_dim)
    model.init_from_train_gaps(train_seq.gaps_hours[1:])  # [0] is the trivial t0 gap
    gap_scale = float(model.gap_scale)
    params = list(model.parameters()) + list(time2vec.parameters())

    def tensors(seq):
        return (torch.tensor(seq.gaps_hours, dtype=torch.float32),
                torch.tensor(seq.calendar_features, dtype=torch.float32))

    tr_gaps, tr_cal = tensors(train_seq)
    train_spans = make_spans(train_seq.n_events, cfg.chunk_len, cfg.burn_in)
    train_rng = np.random.default_rng(cfg.seed)

    if len(train_spans) == 1:
        # Short sequence: one window = the whole sequence, so keep the exact
        # full-sequence objective INCLUDING the censored tail (on a small source
        # that term is a real share of the signal).
        def step_loss(batch):
            return -_nhp_ll(model, time2vec, train_seq, train_rng, gap_scale,
                            cfg.n_mc_samples) / max(train_seq.n_events, 1)
    else:
        def step_loss(batch):
            return _nhp_window_loss(model, time2vec, tr_gaps, tr_cal, batch, gap_scale, cfg.n_mc_samples)

    val_fn = None
    if val_seq.n_events >= 2:
        va_gaps, va_cal = tensors(val_seq)
        val_spans = make_spans(val_seq.n_events, cfg.chunk_len, cfg.burn_in)
        if len(val_spans) == 1:
            def val_fn():  # full validation sequence incl. tail, fixed MC draw
                v = -_nhp_ll(model, time2vec, val_seq, np.random.default_rng(1234),
                             gap_scale, cfg.n_mc_samples).item()
                return v / max(val_seq.n_events, 1)
        else:
            def val_fn():  # windowed, fixed MC draw so the stopping signal is not noise
                gen = torch.Generator().manual_seed(1234)
                total, n = 0.0, 0.0
                for i in range(0, len(val_spans), cfg.batch_size):
                    sp = val_spans[i:i + cfg.batch_size]
                    k = sum(b - s for _, s, b in sp)
                    total += _nhp_window_loss(model, time2vec, va_gaps, va_cal, sp,
                                              gap_scale, cfg.n_mc_samples, gen).item() * k
                    n += k
                return total / max(n, 1)

    fit = _fit(params, [model, time2vec], train_spans, step_loss, val_fn, cfg)

    test_ll = float("nan")
    if test_seq.n_events > 0:
        with torch.no_grad():
            test_ll = _nhp_ll(model, time2vec, test_seq, np.random.default_rng(cfg.seed),
                              gap_scale, cfg.n_mc_samples).item()
    info = {"epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"],
            "n_steps": fit["n_steps"], "hit_cap": fit["hit_cap"], "n_chunks": fit["n_chunks"],
            "n_train": train_seq.n_events, "n_val": val_seq.n_events,
            "best_val_nll_per_event": fit["best"]}
    info["_input_dim"] = time2vec.out_dim + n_cal   # consumed by the runner when checkpointing
    return test_ll, test_seq.n_events, model, time2vec, info


# --------------------------------------------------------------------------
# LogNormMix
# --------------------------------------------------------------------------
def _lnm_ll(model, batch):
    """Full-sequence log-likelihood INCLUDING the tail term (used for test_ll)."""
    return model.log_likelihood(
        batch["inputs"], batch["target_gaps"], batch["mask"],
        batch["lengths"], batch["tail_hours"], batch["terminal_inputs"],
    ).sum()


def _lnm_window_loss(model, inputs_t, targets_t, spans):
    idx, valid, loss_mask = _collate_indices(spans)
    x = inputs_t[idx] * valid.unsqueeze(-1)
    y = torch.where(valid > 0, targets_t[idx], torch.ones(()))   # avoid log(0) on padding
    h = model.forward(x)
    log_p = model.decoder.log_prob(h, y)
    return -(log_p * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def train_lnm_one_source(df, global_study_end, cfg: TrainConfig | None = None):
    """Returns (test_ll, n_predicted_test_gaps, model, info) or None."""
    cfg = cfg or DEFAULT_CONFIG
    _seed(cfg.seed)
    loader = LogNormMixDataLoader(
        df["arrival_timestamp"], use_calendar_features=cfg.use_calendar_features,
        global_study_end=global_study_end,
    )
    splits = loader.get_splits()
    train_seq, val_seq, test_seq = splits["train"], splits["val"], splits["test"]
    if train_seq.n_gaps < 5:
        return None

    input_dim = 1 + train_seq.calendar_features.shape[1]
    model = LogNormMix(input_dim=input_dim, hidden_dim=cfg.hidden_dim, n_components=cfg.n_components)
    model.set_input_normalization(np.log(np.clip(train_seq.gaps_hours, 1e-8, None)))

    def as_tensors(seq):
        inp, tgt = build_step_inputs(seq)  # inputs[j] predicts targets[j]
        return torch.tensor(inp, dtype=torch.float32), torch.tensor(tgt, dtype=torch.float32)

    tr_in, tr_tgt = as_tensors(train_seq)
    train_spans = make_spans(len(tr_tgt), cfg.chunk_len, cfg.burn_in)

    if len(train_spans) == 1:
        # Short sequence: exact full-sequence objective INCLUDING the censored tail.
        train_batch = lnm_pad_batch([train_seq])
        n_train = max(int(train_batch["lengths"].sum()) + 1, 1)

        def step_loss(batch):
            return -_lnm_ll(model, train_batch) / n_train
    else:
        def step_loss(batch):
            return _lnm_window_loss(model, tr_in, tr_tgt, batch)

    val_fn = None
    if val_seq.n_gaps >= 2:
        va_in, va_tgt = as_tensors(val_seq)
        val_spans = make_spans(len(va_tgt), cfg.chunk_len, cfg.burn_in)
        if len(val_spans) == 1:
            val_batch = lnm_pad_batch([val_seq])
            n_val = max(int(val_batch["lengths"].sum()) + 1, 1)

            def val_fn():
                return (-_lnm_ll(model, val_batch) / n_val).item()
        else:
            def val_fn():
                total, n = 0.0, 0
                for i in range(0, len(val_spans), cfg.batch_size):
                    sp = val_spans[i:i + cfg.batch_size]
                    k = sum(b - s for _, s, b in sp)
                    total += _lnm_window_loss(model, va_in, va_tgt, sp).item() * k
                    n += k
                return total / max(n, 1)

    fit = _fit(list(model.parameters()), [model], train_spans, step_loss, val_fn, cfg)

    test_ll, n_pred = float("nan"), 0
    if test_seq.n_gaps >= 2:   # a test split with a single gap has no history step to predict from
        with torch.no_grad():
            tb = lnm_pad_batch([test_seq])
            test_ll = _lnm_ll(model, tb).item()
            n_pred = int(tb["lengths"].sum())
    info = {"epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"],
            "n_steps": fit["n_steps"], "hit_cap": fit["hit_cap"], "n_chunks": fit["n_chunks"],
            "n_train": train_seq.n_gaps, "n_val": val_seq.n_gaps,
            "best_val_nll_per_gap": fit["best"]}
    info["_input_dim"] = input_dim
    return test_ll, n_pred, model, info


# --------------------------------------------------------------------------
# Checkpoints (so plots / re-evaluation never require retraining)
# --------------------------------------------------------------------------
def save_checkpoint(path, kind: str, cfg: TrainConfig, input_dim: int, **modules) -> None:
    """kind is 'lnm' or 'nhp'. `modules` are name -> nn.Module ('model', and
    'time2vec' for the NHP). Normalisation constants (gap_scale, in_mean/in_std)
    are registered buffers, so they are inside the state_dicts."""
    torch.save({"kind": kind, "cfg": cfg.to_dict(), "input_dim": input_dim,
                "state": {k: m.state_dict() for k, m in modules.items()}}, path)


def load_lnm(path):
    """Returns (model, cfg) for a checkpoint written by save_checkpoint."""
    ck = torch.load(path, map_location="cpu")
    assert ck["kind"] == "lnm", f"{path} is a '{ck['kind']}' checkpoint"
    cfg = TrainConfig(**ck["cfg"])
    model = LogNormMix(input_dim=ck["input_dim"], hidden_dim=cfg.hidden_dim, n_components=cfg.n_components)
    model.load_state_dict(ck["state"]["model"])
    return model.eval(), cfg


def load_nhp(path):
    """Returns (model, time2vec, cfg) for a checkpoint written by save_checkpoint."""
    ck = torch.load(path, map_location="cpu")
    assert ck["kind"] == "nhp", f"{path} is a '{ck['kind']}' checkpoint"
    cfg = TrainConfig(**ck["cfg"])
    model = NeuralHawkes(input_dim=ck["input_dim"], hidden_dim=cfg.hidden_dim)
    time2vec = make_time_encoder(cfg)
    model.load_state_dict(ck["state"]["model"])
    time2vec.load_state_dict(ck["state"]["time2vec"])
    return model.eval(), time2vec.eval(), cfg
