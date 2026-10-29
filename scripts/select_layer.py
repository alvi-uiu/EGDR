"""Select the steering depth by development F1 of the static gate (Eq. 9, Sec. 3.2).

The candidate thresholds come from the calibration pool; each layer is then evaluated on the
development panels (``calibration.replicates``), and ACC, F1 and Yes% are means of per-panel
first-token metrics (supplementary Tab. S2). If ``steering.w_tau`` is set, the entropy-gated arm
is also run: its first-token metrics equal the static gate's (Sec. 3.4), and its full-answer
fire rate is the one reported in Tab. S2.
"""

import os

import numpy as np

from egdr import calibration, data, experiment
from egdr.hooks import GateSpec
from egdr.metrics import first_token_metrics
from egdr.modeling import build_prompt, load_model
from egdr.runner import fingerprint, gate_statistics, log, run_arm


def mean_over_panels(rows):
    """Mean of each numeric field over panels; metrics are computed within each panel first."""
    keys = [k for k, v in rows[0].items()
            if k != "panel" and isinstance(v, (int, float)) and not isinstance(v, bool)]
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys}


def main():
    args = experiment.base_parser(__doc__).parse_args()
    cfg = experiment.load_config(args.config)
    out = experiment.output_dir(cfg, args)
    cc, ev, st = cfg["calibration"], cfg["evaluation"], cfg["steering"]
    experiment.seed_everything(cc["seed"])
    alpha = st.get("depth_alpha") or st["alpha"]
    layers = [int(l) for l in st["layer_candidates"]]

    dev_panels = data.calibration_replicates(cfg)
    image_dirs = data.ensure_coco_images(cfg["paths"]["coco_root"],
                                         [q["image"] for p in dev_panels for q in p])
    model, processor = load_model(**cfg["model"])
    directions = {l: experiment.direction_for(cfg, l) for l in layers}
    states = calibration.decision_states(model, processor, dev_panels[0], image_dirs, directions,
                                         ev["pope_max_new_tokens"], log=log)
    w_tau = st.get("w_tau")
    h_lo, h_hi = calibration.entropy_band([r["entropy"] for r in states])

    def run(spec, layer):
        """Per-panel first-token metrics and gate statistics of one arm at one layer."""
        rows = []
        for r, queries in enumerate(dev_panels):
            key = fingerprint(model=cfg["model"]["hf_id"], layer=layer, spec=vars(spec),
                              max_new=ev["pope_max_new_tokens"], records=len(queries))
            recs = run_arm(model, processor, queries,
                           lambda q: build_prompt(processor, q["text"] + data.POPE_SUFFIX),
                           image_dirs, ev["pope_max_new_tokens"], ev["batch_size"], spec,
                           directions[layer],
                           os.path.join(out, "depth", f"layer{layer}-panel{r}-{key}.jsonl"))
            rows.append({"panel": r, **first_token_metrics(recs), **gate_statistics(recs)})
        return rows

    rows = []
    for layer in layers:
        spec = GateSpec(alpha=alpha, tau=calibration.threshold(states, layer, st["k"]))
        per_panel = run(spec, layer)
        rows.append({"layer": layer, "tau": spec.tau, **mean_over_panels(per_panel),
                     "panels": per_panel})
        msg = f"layer {layer}: F1 {rows[-1]['f1']:.2f}  Acc {rows[-1]['acc']:.2f}  tau {spec.tau:.4f}"
        if w_tau:
            gated = GateSpec(alpha=alpha, tau=spec.tau, w_tau=float(w_tau), h_lo=h_lo, h_hi=h_hi)
            g = mean_over_panels(run(gated, layer))
            rows[-1].update({"w_tau": float(w_tau), "f1_w_tau": g["f1"],
                             "fire_rate_w_tau": g["fire_rate"]})
            msg += f"  fire {rows[-1]['fire_rate_w_tau']:.1f}% at w_tau {w_tau}"
        log(msg + f"  ({len(dev_panels)} panels)")

    best = max(rows, key=lambda r: r["f1"])
    data.write_json({"alpha": alpha, "k": st["k"], "n_panels": len(dev_panels),
                     "selected_layer": best["layer"], "layers": rows},
                    os.path.join(out, "depth_selection.json"))
    log(f"selected layer {best['layer']}")


if __name__ == "__main__":
    main()
