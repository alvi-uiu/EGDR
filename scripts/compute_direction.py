"""Estimate the prior-removed visual direction (Eq. 1) at every decoder block."""

import os

import torch

from egdr import calibration, data, experiment
from egdr.direction import compute_directions
from egdr.generation import _open_image
from egdr.modeling import load_model
from egdr.runner import log


def main():
    args = experiment.base_parser(__doc__).parse_args()
    cfg = experiment.load_config(args.config)
    experiment.output_dir(cfg, args)
    experiment.seed_everything(cfg["calibration"]["seed"])
    pool = data.load_pool(cfg["calibration"]["pool"])
    image_dirs = data.ensure_coco_images(cfg["paths"]["coco_root"], [q["image"] for q in pool])

    model, processor = load_model(**cfg["model"])
    states = {r["uid"]: r for r in calibration.decision_states(
        model, processor, pool, image_dirs, {}, cfg["evaluation"]["pope_max_new_tokens"], log=log)}
    answered = [{**q, "answer": states[q["uid"]]["answer"]} for q in pool if q["uid"] in states]
    factual = [q for q in answered if states[q["uid"]]["correct"]]
    erroneous = [q for q in answered if not states[q["uid"]]["correct"]]
    log(f"{len(factual)} correctly and {len(erroneous)} incorrectly answered calibration queries")

    table = compute_directions(model, processor, factual, erroneous,
                               lambda q: _open_image(q, image_dirs))
    table["model"] = cfg["model"]["hf_id"]
    path = cfg["direction"]["file"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(table, path)
    for layer in cfg["steering"]["layer_candidates"]:
        log(f"block {layer}: cos(v_raw, u_prior) = {table['cos_raw_prior'][layer]:.3f}, "
            f"cos(v_raw, v) = {table['cos_raw_direction'][layer]:.3f}, "
            f"cos(v, u_prior) = {table['cos_direction_prior'][layer]:.3f}")


if __name__ == "__main__":
    main()
