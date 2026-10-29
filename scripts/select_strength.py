"""Select alpha and w_tau on captions of the calibration-pool images (Sec. 3.2, Sec. 5.1).

Evaluation images are never used; the selection minimises CHAIR_S, then CHAIR_I.
"""

import itertools
import os

from egdr import data, experiment
from egdr.hooks import GateSpec
from egdr.metrics.chair import CHAIR
from egdr.modeling import build_prompt, load_model
from egdr.runner import fingerprint, gate_statistics, log, run_arm


def main():
    args = experiment.base_parser(__doc__).parse_args()
    cfg = experiment.load_config(args.config)
    out = experiment.output_dir(cfg, args)
    cc, ev = cfg["calibration"], cfg["evaluation"]
    experiment.seed_everything(cc["seed"])
    layer = experiment.steering_layer(cfg, out)
    cal = experiment.load_calibration(out)
    tau = experiment.tau_at(cal, cfg["steering"]["k"])
    h_lo, h_hi = cal["entropy_band"]

    images = data.pool_images(data.load_pool(cc["pool"]))
    coco = cfg["paths"]["coco_root"]
    image_dirs = data.ensure_coco_images(coco, [r["image"] for r in images])
    scorer = CHAIR([r["image_id"] for r in images], data.ensure_coco_annotations(coco))

    model, processor = load_model(**cfg["model"])
    direction = experiment.direction_for(cfg, layer)
    prompt = build_prompt(processor, ev["chair_prompt"])

    rows = []
    for alpha, w_tau in itertools.product(cc["alpha_grid"], cc["w_tau_grid"]):
        spec = GateSpec(alpha=float(alpha), tau=tau, w_tau=float(w_tau), h_lo=h_lo, h_hi=h_hi)
        key = fingerprint(model=cfg["model"]["hf_id"], layer=layer, spec=vars(spec),
                          max_new=ev["chair_max_new_tokens"], records=len(images))
        recs = run_arm(model, processor, images, lambda r: prompt, image_dirs,
                       ev["chair_max_new_tokens"], ev["caption_batch_size"], spec, direction,
                       os.path.join(out, "strength_selection", f"a{alpha}_w{w_tau}-{key}.jsonl"))
        c = scorer.evaluate([{"image_id": r["image_id"], "caption": r["text"]} for r in recs])
        c.pop("per_image")
        rows.append({"alpha": alpha, "w_tau": w_tau, **c, **gate_statistics(recs)})
        log(f"alpha {alpha}, w_tau {w_tau}: C_S {c['chair_s']:.2f}  C_I {c['chair_i']:.2f}")

    best = min(rows, key=lambda r: (r["chair_s"], r["chair_i"]))
    data.write_json({"layer": layer, "tau": tau, "n_images": len(images),
                     "selected": {"alpha": best["alpha"], "w_tau": best["w_tau"]}, "grid": rows},
                    os.path.join(out, "strength_selection.json"))
    log(f"selected alpha {best['alpha']}, w_tau {best['w_tau']}")


if __name__ == "__main__":
    main()
