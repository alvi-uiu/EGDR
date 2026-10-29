"""Calibrate the threshold (Eq. 5) and entropy band (Eq. 6) at the locked layer."""

import os

import numpy as np

from egdr import calibration, data, experiment
from egdr.modeling import eos_token_ids, load_model
from egdr.runner import log


def main():
    p = experiment.base_parser(__doc__)
    p.add_argument("--generation-band", action="store_true",
                   help="also estimate the band from free-running generation (Tab. 3 ablation)")
    args = p.parse_args()
    cfg = experiment.load_config(args.config)
    out = experiment.output_dir(cfg, args)
    cc, ev = cfg["calibration"], cfg["evaluation"]
    experiment.seed_everything(cc["seed"])

    layer = experiment.steering_layer(cfg, out)
    direction = experiment.direction_for(cfg, layer)
    queries = data.load_pool(cc["pool"])
    image_dirs = data.ensure_coco_images(cfg["paths"]["coco_root"], [q["image"] for q in queries])

    model, processor = load_model(**cfg["model"])
    raw = experiment.direction_for(cfg, layer, projected=False)
    states = calibration.decision_states(model, processor, queries, image_dirs,
                                         {"v": direction, "v_raw": raw},
                                         ev["pope_max_new_tokens"], log=log)
    fact = [r["risk"]["v"] for r in states if r["correct"]]
    tau = calibration.threshold(states, "v", cfg["steering"]["k"])
    band = calibration.entropy_band([r["entropy"] for r in states])
    answer_steps = [e for r in states for e in r["step_entropies"]]
    result = {
        "model": cfg["model"]["hf_id"],
        "layer": layer,
        "k": cfg["steering"]["k"],
        "tau": tau,
        "entropy_band": list(band),
        "n_queries": len(states),
        "n_factual": len(fact),
        "factual_risks": fact,
        "factual_risks_unprojected": [r["risk"]["v_raw"] for r in states if r["correct"]],
        "answer_steps_above_h_hi": 100.0 * float(np.mean(np.array(answer_steps) > band[1])),
        "direction_file": cfg["direction"]["file"],
        "direction_sha256": experiment.sha256(cfg["direction"]["file"]),
    }
    log(f"layer {layer}: tau = {tau:.4f} (k = {result['k']}, {len(fact)}/{len(states)} factual), "
        f"band = [{band[0]:.4f}, {band[1]:.4f}]")

    if args.generation_band:
        images = data.pool_images(queries)
        ent = calibration.generation_entropies(model, processor, images, image_dirs,
                                               ev["chair_prompt"], ev["chair_max_new_tokens"],
                                               eos_token_ids(model, processor), log=log)
        gband = calibration.entropy_band(ent)
        result["generation_band"] = list(gband)
        result["generation_band_steps"] = len(ent)
        log(f"generation band = [{gband[0]:.4f}, {gband[1]:.4f}] over {len(ent)} steps")
        result["caption_steps_above_h_hi"] = 100.0 * float(np.mean(np.array(ent) > band[1]))

    data.write_json(result, os.path.join(out, "calibration.json"))


if __name__ == "__main__":
    main()
