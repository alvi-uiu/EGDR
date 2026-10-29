"""Forward hooks implementing Algorithm S1: the static gate (w_tau = 0) and EGDR on a frozen model.

At every decoding step the gate acts on the current state (the last position of the pass).
The steering hook on block l* runs before lm_head, so the entropy it reads comes from the
previous step; at the first generated token e_hat = 0 and tau_eff = tau.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from .gate import effective_threshold, normalized_entropy, rescale_entropy, risk_score
from .modeling import decoder_layers, lm_head


@dataclass
class GateSpec:
    """Operating point of one gated arm; ``w_tau = 0`` is the static gate."""

    alpha: float
    tau: float
    w_tau: float = 0.0
    h_lo: float = 0.0
    h_hi: float = 1.0


class EntropyTracker:
    """Normalised entropy of the most recent next-token distribution, per batch row."""

    def __init__(self, model):
        self.value = None
        self.history = []
        self.n_nonfinite = 0
        self._handle = lm_head(model).register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        with torch.no_grad():
            ent, n_bad = normalized_entropy(output[:, -1, :])
        self.n_nonfinite += n_bad
        self.value = ent
        self.history.append(ent.detach().cpu().numpy().copy())

    def reset(self):
        self.value = None
        self.history = []

    def remove(self):
        self._handle.remove()


class GatedSteeringHook:
    """Adds ``alpha * v`` to the current state of block l* when its risk exceeds tau_eff (Eq. 8)."""

    def __init__(self, model, direction, spec: GateSpec, tracker: EntropyTracker = None):
        if spec.w_tau > 0 and tracker is None:
            raise ValueError("an entropy-gated arm needs an EntropyTracker")
        device = next(model.parameters()).device
        self.spec = spec
        self.tracker = tracker
        self.v_unit = direction.unit.to(device)
        self.v_inject = direction.vector.float().to(device)
        self.tel = defaultdict(list)
        self._handle = decoder_layers(model)[direction.layer].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        with torch.no_grad():
            h = hidden[:, -1, :].float()
            b = h.shape[0]
            risk = risk_score(h, self.v_unit)

            prev = self.tracker.value if self.tracker is not None else None
            if self.spec.w_tau > 0 and prev is not None and prev.shape[0] == b:
                e_hat = rescale_entropy(prev.to(h.device), self.spec.h_lo, self.spec.h_hi)
            else:
                e_hat = torch.zeros(b, device=h.device)
            tau_eff = effective_threshold(self.spec.tau, self.spec.w_tau, e_hat)

            fired = risk > tau_eff
            delta = (fired.float() * self.spec.alpha).unsqueeze(-1) * self.v_inject.view(1, -1)
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            steered = hidden.clone()
            steered[:, -1, :] = (h + delta).to(hidden.dtype)

            self.tel["e_hat"].append(e_hat.cpu().numpy())
            self.tel["tau_eff"].append(tau_eff.cpu().numpy())
            self.tel["risk"].append(risk.cpu().numpy())
            self.tel["fired"].append(fired.cpu().numpy().astype(np.uint8))

        if isinstance(output, tuple):
            return (steered,) + tuple(output[1:])
        return steered

    def reset(self):
        self.tel = defaultdict(list)

    def row_telemetry(self, row: int, n_steps: int):
        """Telemetry of one batch row over its first ``n_steps`` decoding steps."""
        out = {k: [float(a[row]) for a in v[:n_steps]] for k, v in self.tel.items()}
        out["fired"] = [int(x) for x in out.get("fired", [])]
        return out

    def remove(self):
        self._handle.remove()


class Steering:
    """Context manager installing the hooks of one arm; ``spec=None`` leaves the model unsteered."""

    def __init__(self, model, direction=None, spec: GateSpec = None):
        self.model = model
        self.direction = direction
        self.spec = spec
        self.tracker = None
        self.hook = None

    def __enter__(self):
        if self.spec is not None:
            if self.spec.w_tau > 0:
                self.tracker = EntropyTracker(self.model)
            self.hook = GatedSteeringHook(self.model, self.direction, self.spec, self.tracker)
        return self

    def reset(self):
        """Call before every ``generate``."""
        if self.tracker is not None:
            self.tracker.reset()
        if self.hook is not None:
            self.hook.reset()

    def __exit__(self, *exc):
        if self.hook is not None:
            self.hook.remove()
        if self.tracker is not None:
            self.tracker.remove()
        return False
