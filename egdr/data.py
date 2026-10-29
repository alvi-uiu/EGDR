"""Calibration data: the existence-query pool and the COCO images and annotations it uses."""

import json
import os
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

COCO_IMAGE_URL = "http://images.cocodataset.org/{split}/{name}"
COCO_ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
POPE_SUFFIX = " Please answer yes or no."


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def fetch(url, dest, timeout=120):
    """Download ``url`` to ``dest`` unless it already exists."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def coco_split_of(file_name):
    return "train2014" if "train2014" in file_name else "val2014"


def ensure_coco_images(coco_root, file_names, workers=16):
    """Download the COCO 2014 images in ``file_names`` that are not yet present."""
    names = sorted(set(file_names))
    missing = [n for n in names
               if not os.path.exists(os.path.join(coco_root, coco_split_of(n), n))]

    def grab(name):
        split = coco_split_of(name)
        try:
            fetch(COCO_IMAGE_URL.format(split=split, name=name),
                  os.path.join(coco_root, split, name), timeout=60)
            return True
        except OSError:
            return False

    if missing:
        with ThreadPoolExecutor(workers) as ex:
            ok = sum(ex.map(grab, missing))
        if ok < len(missing):
            raise RuntimeError(f"{len(missing) - ok} COCO images could not be downloaded")
    return [os.path.join(coco_root, "val2014"), os.path.join(coco_root, "train2014")]


def ensure_coco_annotations(coco_root):
    """Instance and caption annotations of COCO 2014, used by CHAIR."""
    ann = os.path.join(coco_root, "annotations")
    need = ("instances_train2014.json", "instances_val2014.json",
            "captions_train2014.json", "captions_val2014.json")
    if not all(os.path.exists(os.path.join(ann, n)) for n in need):
        archive = os.path.join(coco_root, "annotations_trainval2014.zip")
        fetch(COCO_ANNOTATION_URL, archive, timeout=1800)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(coco_root)
    return ann


def build_calibration_pool(instances_train_path, n_images=100, seed=42):
    """Calibration pool: n_images COCO train2014 images with five POPE-style existence queries each.

    Images alternate between 3 present + 2 absent and 2 present + 3 absent objects, so the
    pool is balanced; absent objects are drawn uniformly from the categories not in the image.
    """
    inst = read_json(instances_train_path)
    names = {c["id"]: c["name"] for c in inst["categories"]}
    files = {i["id"]: i["file_name"] for i in inst["images"]}
    present = {}
    for a in inst["annotations"]:
        if not a.get("iscrowd"):
            present.setdefault(a["image_id"], set()).add(names[a["category_id"]])
    rng = random.Random(seed)
    candidates = sorted(i for i, objs in present.items() if len(objs) >= 3)
    rng.shuffle(candidates)
    queries = []
    for k, image_id in enumerate(sorted(candidates[:n_images])):
        n_yes = 3 if k % 2 == 0 else 2
        yes = rng.sample(sorted(present[image_id]), n_yes)
        no = rng.sample(sorted(set(names.values()) - present[image_id]), 5 - n_yes)
        for obj, label in [(o, "yes") for o in yes] + [(o, "no") for o in no]:
            queries.append({"uid": f"cal_{len(queries)}", "image_id": image_id,
                            "image": files[image_id], "object": obj,
                            "text": f"Is there a {obj} in the image?", "label": label})
    return queries


def load_pool(path):
    """Calibration queries (COCO train2014, disjoint from every evaluation split)."""
    return read_json(path)["queries"]


def calibration_replicates(cfg):
    """Development panels (Sec. 5.1): ``calibration.replicates`` pools built with seeds
    ``seed + r``; replicate 0 is the calibration pool. Built pools are cached next to it."""
    cc = cfg["calibration"]
    pool = read_json(cc["pool"])
    panels = [pool["queries"]]
    stem = os.path.splitext(cc["pool"])[0]
    for r in range(1, int(cc.get("replicates", 1))):
        path = f"{stem}_r{r}.json"
        if not os.path.exists(path):
            ann = ensure_coco_annotations(cfg["paths"]["coco_root"])
            seed = int(pool["seed"]) + r
            queries = build_calibration_pool(os.path.join(ann, "instances_train2014.json"),
                                             int(pool["n_images"]), seed)
            write_json({"split": "train2014", "seed": seed, "n_images": int(pool["n_images"]),
                        "queries": queries}, path)
        panels.append(read_json(path)["queries"])
    return panels


def pool_images(queries):
    """Distinct images of the calibration pool as caption records."""
    seen = {}
    for q in queries:
        seen.setdefault(q["image_id"], q["image"])
    return [{"uid": str(i), "image_id": i, "image": n} for i, n in sorted(seen.items())]
