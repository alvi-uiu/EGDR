"""Prior-removed visual steering direction (Eq. 1; supplementary Sec. A.1)."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import POPE_SUFFIX
from .modeling import build_prompt, build_text_prompt, decoder_layers


@dataclass
class SteeringDirection:
    """Unit direction for decoder block ``layer``; the gate injects ``alpha * vector``."""

    vector: torch.Tensor
    layer: int

    @property
    def unit(self) -> torch.Tensor:
        v = self.vector.float()
        return v / (v.norm() + 1e-8)


def load_direction(path: str, layer: int, projected: bool = True) -> SteeringDirection:
    """Direction of decoder block ``layer`` from ``compute_direction.py``; ``projected=False`` gives v_raw."""
    table = torch.load(path, map_location="cpu", weights_only=True)
    key = "vectors" if projected else "raw_vectors"
    return SteeringDirection(vector=table[key][layer].float(), layer=layer)


@torch.no_grad()
def answer_states(model, processor, image, question, answer):
    """Hidden states of every decoder block at the last answer token, shape (L, D).

    With ``image=None`` the answer is replayed with the visual tokens removed. States are the
    raw block outputs, the tensors the steering hook edits (``output_hidden_states`` applies the
    final norm to the last block).
    """
    if image is None:
        text = build_text_prompt(processor, question + POPE_SUFFIX) + " " + answer
        inputs = processor.tokenizer(text, return_tensors="pt").to(model.device)
    else:
        text = build_prompt(processor, question + POPE_SUFFIX) + " " + answer
        inputs = processor(images=image, text=text, return_tensors="pt").to(model.device)
    layers = decoder_layers(model)
    states = [None] * len(layers)

    def grab(i):
        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            states[i] = h[0, -1].float().cpu()
        return hook

    handles = [layer.register_forward_hook(grab(i)) for i, layer in enumerate(layers)]
    try:
        model(**inputs)
    finally:
        for h in handles:
            h.remove()
    return torch.stack(states)


@torch.no_grad()
def compute_directions(model, processor, factual, erroneous, image_loader):
    """Eq. 1 at every decoder block.

    Args:
        factual: correctly answered calibration queries, each with the model's ``answer``.
        erroneous: incorrectly answered calibration queries, each with the model's ``answer``.
        image_loader: maps a query to its PIL image.
    """
    raw, h_fact, h_err = [], [], []
    for q in factual:
        image = image_loader(q)
        if image is None:
            continue
        h_txt = answer_states(model, processor, None, q["text"], q["answer"])
        h_img = answer_states(model, processor, image, q["text"], q["answer"])
        raw.append(h_img - h_txt)
        h_fact.append(h_txt)
    for q in erroneous:
        h_err.append(answer_states(model, processor, None, q["text"], q["answer"]))
    if not raw or not h_err:
        raise RuntimeError("both correctly and incorrectly answered queries are required")
    v_raw = torch.stack(raw).mean(dim=0)                                            # (L, D)
    u = F.normalize(torch.stack(h_err).mean(dim=0) - torch.stack(h_fact).mean(dim=0), dim=-1)
    v = F.normalize(v_raw - (v_raw * u).sum(dim=-1, keepdim=True) * u, dim=-1)
    v_raw_unit = F.normalize(v_raw, dim=-1)
    return {
        "vectors": {l: v[l].clone() for l in range(v.shape[0])},
        "raw_vectors": {l: v_raw_unit[l].clone() for l in range(v.shape[0])},
        "cos_raw_prior": F.cosine_similarity(v_raw, u, dim=-1).tolist(),
        "cos_raw_direction": F.cosine_similarity(v_raw, v, dim=-1).tolist(),
        "cos_direction_prior": F.cosine_similarity(v, u, dim=-1).tolist(),
        "n_factual": len(raw),
        "n_erroneous": len(h_err),
    }
