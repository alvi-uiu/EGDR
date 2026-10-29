"""Gate equations of EGDR as pure tensor operations (equation numbers follow the paper)."""

import math

import torch
import torch.nn.functional as F


def risk_score(hidden: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Eq. 2: R(h) = -cos(h, v) over the last dimension of ``hidden``."""
    h = hidden.float()
    v = direction.to(device=h.device, dtype=torch.float32).view(*([1] * (h.dim() - 1)), -1)
    return -F.cosine_similarity(h, v, dim=-1)


def normalized_entropy(logits: torch.Tensor):
    """Eq. 4: H(p) / log|V| in [0, 1] per row; returns (entropy, number of non-finite rows)."""
    z = logits.float()
    n_bad = int((~torch.isfinite(z).all(dim=-1)).sum())
    if n_bad:
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=-1e4)
    p = torch.softmax(z, dim=-1)
    ent = -(p * torch.log(p + 1e-12)).sum(dim=-1) / math.log(z.shape[-1])
    return torch.nan_to_num(ent, nan=0.0).clamp_(0.0, 1.0), n_bad


def rescale_entropy(entropy: torch.Tensor, h_lo: float, h_hi: float) -> torch.Tensor:
    """Eq. 6: clip((H - H_lo) / (H_hi - H_lo), 0, 1)."""
    width = max(float(h_hi) - float(h_lo), 1e-6)
    return ((entropy.float() - float(h_lo)) / width).clamp(0.0, 1.0)


def effective_threshold(tau: float, w_tau: float, e_hat: torch.Tensor) -> torch.Tensor:
    """Eq. 7: tau - w_tau * e_hat * |tau|, which lowers the threshold for either sign of tau."""
    if w_tau < 0:
        raise ValueError("w_tau must be non-negative")
    return float(tau) - float(w_tau) * e_hat.float() * abs(float(tau))

