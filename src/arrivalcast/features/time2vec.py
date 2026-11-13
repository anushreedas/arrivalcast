"""
Time2Vec encoding (Kazemi et al., 2019).

Time2Vec is a *learned* embedding of a scalar time value: one linear
component plus (k-1) periodic components with learned frequency/phase,
trained jointly with the downstream model. Unlike calendar_features.py
(fixed, hand-built sin/cos features), Time2Vec lets the model discover
whatever periodicities are useful directly from data.

Like calendar features, this only makes sense for the neural models
(Neural Hawkes, LogNormMix) — the classical Poisson/Hawkes baselines have
no learned parameters that could consume an embedding like this.

Usage: feed it either the raw inter-arrival gap Δt, or an absolute elapsed
time since the observation window start, at each event step; concatenate
the output with whatever other per-step inputs your model uses (see
models/neural_hawkes/dataloader.py and models/lognormmix/dataloader.py).

Requires PyTorch (not needed by core/, evaluation/ or features/calendar_features.py).
"""
from __future__ import annotations

import torch
from torch import nn


class Time2Vec(nn.Module):
    """
    Time2Vec(t) = [w0*t + b0, sin(w1*t + b1), ..., sin(w_{k-1}*t + b_{k-1})]

    Parameters
    ----------
    out_dim : int
        Total embedding dimension k. One dimension is linear, the
        remaining (k - 1) are periodic (sine).
    """

    def __init__(self, out_dim: int = 8):
        super().__init__()
        if out_dim < 2:
            raise ValueError("out_dim must be >= 2 (1 linear + >=1 periodic term)")
        self.out_dim = out_dim
        self.linear_weight = nn.Parameter(torch.randn(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        n_periodic = out_dim - 1
        self.periodic_weight = nn.Parameter(torch.randn(n_periodic))
        self.periodic_bias = nn.Parameter(torch.zeros(n_periodic))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : tensor of shape (...,) or (..., 1) — scalar time values
            (e.g. Δt in hours, or elapsed time since window start).

        Returns
        -------
        tensor of shape (..., out_dim)

        Note: a 0-D or 1-D input is ALWAYS unsqueezed to add a trailing
        feature dim, regardless of its size. Inferring this from
        `t.shape[-1] != 1` (as an earlier version of this function did) is
        ambiguous for a length-1 batch: a 1-D tensor of shape (1,) (one
        event, no feature dim yet) is indistinguishable by size alone from
        an already-correct (..., 1)-shaped tensor, so that check silently
        skipped the unsqueeze exactly when there's a single event in a
        batch (e.g. a source whose test split has exactly one event) and
        returned a 1-D tensor instead of 2-D, breaking concatenation with
        (always 2-D) calendar features downstream. Using `t.dim()` instead
        of `t.shape[-1]` removes the ambiguity: dimensionality, not size,
        is what actually distinguishes "needs a feature dim" from
        "already has one."
        """
        if t.dim() <= 1:
            t = t.unsqueeze(-1)

        linear = self.linear_weight * t + self.linear_bias  # (..., 1)
        periodic = torch.sin(t * self.periodic_weight + self.periodic_bias)  # (..., k-1)
        return torch.cat([linear, periodic], dim=-1)


class RawGapEncoding(nn.Module):
    """Ablation stand-in for Time2Vec: the (already rescaled) gap enters the
    model as ONE raw feature, with no learned parameters and no periodic terms.

    Drop-in replacement (same call signature, `out_dim` attribute) so the rest of
    the pipeline does not need to know which encoder is in use. Compared with
    Time2Vec this removes the learned sine components and the learned
    scale/shift of the linear term (the recurrent layer's own input weights can
    still rescale a raw feature, so the linear part is not really lost).
    """

    out_dim = 1

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() <= 1:
            t = t.unsqueeze(-1)
        return t
