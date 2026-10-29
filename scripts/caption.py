"""Describe one image under unsteered decoding, the static gate and EGDR at the locked operating point."""

from egdr import experiment
from egdr.generation import generate_batch
from egdr.hooks import Steering
from egdr.modeling import build_prompt, load_model
from PIL import Image


def main():
    p = experiment.base_parser(__doc__)
    p.add_argument("--image", required=True, help="path to an image")
    p.add_argument("--prompt", default=None, help="defaults to evaluation.chair_prompt")
    p.add_argument("--arms", nargs="+", default=["regular", "static", "egdr"],
                   help="regular, static, egdr; overrides as egdr:w_tau=0.25")
    p.add_argument("--max-new-tokens", type=int, default=None)
    args = p.parse_args()
    cfg = experiment.load_config(args.config)
    out = experiment.output_dir(cfg, args)
    ev = cfg["evaluation"]
    experiment.seed_everything(cfg["calibration"]["seed"])

    model, processor = load_model(**cfg["model"])
    prompt = build_prompt(processor, args.prompt or ev["chair_prompt"])
    image = Image.open(args.image).convert("RGB")
    max_new = args.max_new_tokens or ev["chair_max_new_tokens"]

    for arm in args.arms:
        spec, direction = None, None
        if not arm.startswith("regular"):
            layer = experiment.steering_layer(cfg, out)
            spec = experiment.gate_spec(cfg, experiment.load_calibration(out), out, arm)
            direction = experiment.arm_direction(cfg, layer, arm)
        with Steering(model, direction, spec) as st:
            row = generate_batch(model, processor, [image], [prompt], max_new,
                                 st if spec is not None else None)[0]
        fired = row.get("fired")
        rate = f", gate open at {100.0 * sum(fired) / max(1, len(fired)):.1f}% of steps" if fired else ""
        print(f"[{arm}] ({row['n_tokens']} tokens{rate})\n{row['text']}\n")


if __name__ == "__main__":
    main()
