"""Configuration, operating-point resolution and arm construction shared by the scripts."""

import argparse
import hashlib
import json
import os
import random

import numpy as np
import yaml

from .direction import load_direction
from .hooks import GateSpec

ARMS = ("regular", "static", "static_matched", "egdr", "egdr_genband", "egdr_unprojected")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="YAML file in configs/")
    p.add_argument("--output-dir", default=None, help="overrides paths.output_dir")
    return p


def output_dir(cfg, args=None):
    d = (args.output_dir if args is not None and args.output_dir else None) \
        or cfg["paths"]["output_dir"]
    os.makedirs(d, exist_ok=True)
    return d


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def steering_layer(cfg, out_dir):
    """Locked depth: from the config, else from ``depth_selection.json`` (Eq. 9)."""
    layer = cfg["steering"].get("layer")
    if layer is None:
        sel = _read(os.path.join(out_dir, "depth_selection.json"))
        if sel is None:
            raise SystemExit("steering.layer is not set; run scripts/select_layer.py first")
        layer = sel["selected_layer"]
    return int(layer)


def strength_and_weight(cfg, out_dir):
    """Locked ``(alpha, w_tau)``: from the config, else from ``strength_selection.json``."""
    alpha, w_tau = cfg["steering"].get("alpha"), cfg["steering"].get("w_tau")
    if alpha is None or w_tau is None:
        sel = _read(os.path.join(out_dir, "strength_selection.json"))
        if sel is None:
            raise SystemExit("steering.alpha / steering.w_tau are not set; "
                             "run scripts/select_strength.py first")
        alpha = sel["selected"]["alpha"] if alpha is None else alpha
        w_tau = sel["selected"]["w_tau"] if w_tau is None else w_tau
    return float(alpha), float(w_tau)


def load_calibration(out_dir):
    cal = _read(os.path.join(out_dir, "calibration.json"))
    if cal is None:
        raise SystemExit("calibration.json not found; run scripts/calibrate.py first")
    return cal


def tau_at(calibration, k, key="factual_risks"):
    """Threshold at quantile ``k`` of the stored factual-state risks (Eq. 5)."""
    return float(np.percentile(calibration[key], 100.0 * float(k)))


def direction_for(cfg, layer, projected=True):
    path = cfg["direction"]["file"]
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run scripts/compute_direction.py first")
    return load_direction(path, layer, projected)


def arm_direction(cfg, layer, spec):
    """Direction used by an arm: v_raw for ``egdr_unprojected``, the prior-removed v otherwise."""
    return direction_for(cfg, layer, projected=not spec.startswith("egdr_unprojected"))


def parse_arm(spec):
    """``"egdr:w_tau=0.25,alpha=0.6"`` -> ``("egdr", {"w_tau": 0.25, "alpha": 0.6})``."""
    name, _, rest = spec.partition(":")
    if name not in ARMS:
        raise SystemExit(f"unknown arm '{name}'; choose from {', '.join(ARMS)}")
    overrides = {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        if key not in ("alpha", "w_tau", "k"):
            raise SystemExit(f"unknown override '{key}' in arm '{spec}'")
        overrides[key] = float(value)
    return name, overrides


def gate_spec(cfg, calibration, out_dir, spec):
    """GateSpec of an arm (``None`` for ``regular``); arms are described in the README."""
    name, ov = parse_arm(spec)
    if name == "regular":
        return None
    alpha, w_tau = strength_and_weight(cfg, out_dir)
    alpha = ov.get("alpha", alpha)
    st = cfg["steering"]
    if name == "static":
        k = ov.get("k", st["k"])
        return GateSpec(alpha=alpha, tau=tau_at(calibration, k))
    if name == "static_matched":
        k = ov.get("k", st.get("k_matched"))
        if k is None:
            raise SystemExit("steering.k_matched is not set in the configuration")
        return GateSpec(alpha=alpha, tau=tau_at(calibration, k))
    k = ov.get("k", st["k"])
    w = ov.get("w_tau", w_tau)
    band = calibration.get("generation_band") if name == "egdr_genband" else calibration["entropy_band"]
    if band is None:
        raise SystemExit("generation_band missing; run scripts/calibrate.py --generation-band")
    key = "factual_risks_unprojected" if name == "egdr_unprojected" else "factual_risks"
    return GateSpec(alpha=alpha, tau=tau_at(calibration, k, key), w_tau=w, h_lo=band[0], h_hi=band[1])


def arm_tag(spec):
    """File-system-safe name of an arm specification."""
    return spec.replace(":", "__").replace(",", "_").replace("=", "")
