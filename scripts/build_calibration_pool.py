"""Build the calibration pool: 100 COCO train2014 images with 500 balanced existence queries."""

import argparse

from egdr import data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco-root", default="data/coco")
    p.add_argument("--images", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/calibration/pool.json")
    args = p.parse_args()
    ann = data.ensure_coco_annotations(args.coco_root)
    queries = data.build_calibration_pool(f"{ann}/instances_train2014.json", args.images, args.seed)
    data.write_json({"split": "train2014", "seed": args.seed, "n_images": args.images,
                     "queries": queries}, args.output)
    print(f"{len(queries)} queries on {args.images} images -> {args.output}")


if __name__ == "__main__":
    main()
