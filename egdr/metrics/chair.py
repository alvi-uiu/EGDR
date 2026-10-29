"""CHAIR metrics (Rohrbach et al., EMNLP 2018): object mentions are matched to COCO synonyms and
checked against the instance annotations and reference captions of each image."""

import json
import os

import inflect
import nltk

_INFLECT = inflect.engine()
SYNONYMS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synonyms.txt")


def singularize(word):
    res = _INFLECT.singular_noun(word)
    return res if res else word


def ensure_nltk_tokenizer():
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


class CHAIR:
    def __init__(self, image_ids, annotation_dir, synonyms_file=SYNONYMS_FILE):
        ensure_nltk_tokenizer()
        self.imid_to_objects = {int(i): set() for i in image_ids}
        self.annotation_dir = annotation_dir

        with open(synonyms_file) as f:
            synonyms = [line.strip().split(", ") for line in f if line.strip()]
        self.mscoco_objects = []
        self.inverse_synonym_dict = {}
        for group in synonyms:
            self.mscoco_objects.extend(group)
            for s in group:
                self.inverse_synonym_dict[s] = group[0]

        double_words = [
            "motor bike", "motor cycle", "air plane", "traffic light", "stop sign",
            "parking meter", "sports ball", "baseball bat", "baseball glove",
            "tennis racket", "wine glass", "hot dog", "cell phone", "teddy bear",
            "hair drier", "potted plant",
        ]
        self.double_word_dict = {w: w for w in double_words}
        for animal in ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
                       "bear", "zebra", "giraffe", "animal", "cub"):
            self.double_word_dict[f"baby {animal}"] = animal
            self.double_word_dict[f"adult {animal}"] = animal
        for vehicle in ("jet", "train"):
            self.double_word_dict[f"passenger {vehicle}"] = vehicle
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"
        self._load_annotations()

    def caption_to_words(self, caption):
        """Return ``(words, node_words)``: COCO object mentions and their canonical names."""
        words = [singularize(w) for w in nltk.word_tokenize(caption.lower())]
        i, merged = 0, []
        while i < len(words):
            pair = " ".join(words[i:i + 2])
            if pair in self.double_word_dict:
                merged.append(self.double_word_dict[pair])
                i += 2
            else:
                merged.append(words[i])
                i += 1
        words = [w for w in merged if w in self.mscoco_objects]
        node_words = [self.inverse_synonym_dict[w] for w in words]
        return words, node_words

    def _load_annotations(self):
        def load(name):
            with open(os.path.join(self.annotation_dir, name)) as f:
                return json.load(f)

        inst_train = load("instances_train2014.json")
        inst_val = load("instances_val2014.json")
        id_to_name = {c["id"]: c["name"] for c in inst_train["categories"]}
        for ann in inst_train["annotations"] + inst_val["annotations"]:
            if ann["image_id"] in self.imid_to_objects:
                self.imid_to_objects[ann["image_id"]].add(
                    self.inverse_synonym_dict[id_to_name[ann["category_id"]]])
        for name in ("captions_train2014.json", "captions_val2014.json"):
            for ann in load(name)["annotations"]:
                if ann["image_id"] in self.imid_to_objects:
                    _, nodes = self.caption_to_words(ann["caption"])
                    self.imid_to_objects[ann["image_id"]].update(nodes)

    def evaluate(self, captions):
        """CHAIR_S, CHAIR_I (percent) and per-image detail.

        Args:
            captions: list of ``{"image_id": int, "caption": str}``.
        """
        n_caps = n_hal_caps = n_obj = n_hal_obj = 0
        per_image = []
        for c in captions:
            imid = int(c["image_id"])
            if imid not in self.imid_to_objects:
                continue
            _, nodes = self.caption_to_words(c["caption"])
            gt = self.imid_to_objects[imid]
            hallucinated = [w for w in nodes if w not in gt]
            n_caps += 1
            n_obj += len(nodes)
            n_hal_obj += len(hallucinated)
            n_hal_caps += int(bool(hallucinated))
            per_image.append({"image_id": imid, "mentioned": nodes,
                              "hallucinated": hallucinated})
        return {
            "chair_s": 100.0 * n_hal_caps / max(1, n_caps),
            "chair_i": 100.0 * n_hal_obj / max(1, n_obj),
            "n_images": n_caps,
            "n_mentions": n_obj,
            "n_hallucinated": n_hal_obj,
            "words": (sum(len(c["caption"].split()) for c in captions) / max(1, len(captions))),
            "per_image": per_image,
        }
