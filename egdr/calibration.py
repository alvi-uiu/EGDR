"""Threshold (Eq. 5) and entropy band (Eq. 6), measured on unsteered calibration-pool queries."""

import numpy as np
import torch

from .data import POPE_SUFFIX
from .gate import risk_score
from .generation import _open_image
from .hooks import EntropyTracker
from .metrics.pope import parse_yes_no
from .modeling import build_prompt, decoder_layers, eos_token_ids


@torch.no_grad()
def decision_states(model, processor, queries, image_dirs, directions, max_new_tokens=10,
                    log=print):
    """Risk under each named direction and entropy at the answer-decision state (last prompt
    position) of each existence query, with answer correctness and per-step answer entropies."""
    probes, handles = {}, []
    layers = decoder_layers(model)
    for name, d in directions.items():
        unit = d.unit.to(next(model.parameters()).device)

        def hook(module, inputs, output, name=name, unit=unit):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] > 1:
                probes[name] = float(risk_score(h[:, -1], unit)[0])
        handles.append(layers[d.layer].register_forward_hook(hook))
    tracker = EntropyTracker(model)
    eos = eos_token_ids(model, processor)

    records = []
    try:
        for i, q in enumerate(queries):
            image = _open_image(q, image_dirs)
            if image is None:
                log(f"missing image, skipped: {q['image']}")
                continue
            prompt = build_prompt(processor, q["text"] + POPE_SUFFIX)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            probes.clear()
            tracker.reset()
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 num_beams=1, pad_token_id=processor.tokenizer.pad_token_id)
            ids = out[0, inputs["input_ids"].shape[1]:].tolist()
            n = next((j + 1 for j, t in enumerate(ids) if t in eos), len(ids))
            text = processor.tokenizer.decode(ids, skip_special_tokens=True)
            pred = parse_yes_no(text)
            label = str(q["label"]).lower()
            records.append({
                "uid": q["uid"], "risk": dict(probes),
                "entropy": float(tracker.history[0][0]),
                "step_entropies": [float(a[0]) for a in tracker.history[:n]],
                "answer": processor.tokenizer.decode(ids[:n], skip_special_tokens=True).strip(),
                "pred": pred, "label": label, "correct": pred == label,
            })
            if (i + 1) % 100 == 0:
                log(f"  calibration {i + 1}/{len(queries)}")
    finally:
        for h in handles:
            h.remove()
        tracker.remove()
    return records


def threshold(records, name, k):
    """Eq. 5: Percentile(S_fact, k) over the correctly answered queries."""
    fact = [r["risk"][name] for r in records if r["correct"]]
    if not fact:
        raise RuntimeError("no correctly answered calibration query")
    return float(np.percentile(fact, 100.0 * k))


def entropy_band(entropies, lo=10.0, hi=90.0):
    """(H_lo, H_hi): 10th and 90th percentiles of normalised entropy."""
    h_lo, h_hi = float(np.percentile(entropies, lo)), float(np.percentile(entropies, hi))
    if h_hi - h_lo < 1e-6:
        h_hi = h_lo + 1e-6
    return h_lo, h_hi


@torch.no_grad()
def generation_entropies(model, processor, records, image_dirs, prompt, max_new_tokens,
                         eos_ids, log=print):
    """Entropy at every step of unsteered free-running generation (band ablation, Tab. 3)."""
    tracker = EntropyTracker(model)
    values = []
    try:
        chat = build_prompt(processor, prompt)
        for i, rec in enumerate(records):
            image = _open_image(rec, image_dirs)
            if image is None:
                continue
            inputs = processor(images=image, text=chat, return_tensors="pt").to(model.device)
            tracker.reset()
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 num_beams=1, pad_token_id=processor.tokenizer.pad_token_id)
            ids = out[0, inputs["input_ids"].shape[1]:].tolist()
            n = next((j + 1 for j, t in enumerate(ids) if t in eos_ids), len(ids))
            values.extend(float(a[0]) for a in tracker.history[:n])
            if (i + 1) % 25 == 0:
                log(f"  generation band {i + 1}/{len(records)}")
    finally:
        tracker.remove()
    return values
