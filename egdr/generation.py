"""Batched greedy generation with per-sample telemetry and resumable output."""

import gc
import json
import os
import time
from collections import defaultdict

import torch
from PIL import Image

from .modeling import eos_token_ids


def generated_length(token_ids, eos_ids):
    """Number of tokens up to and including the first end-of-sequence token."""
    for i, t in enumerate(token_ids):
        if int(t) in eos_ids:
            return i + 1
    return len(token_ids)


def _length_buckets(processor, records, prompts):
    """Group prompts by token length so that batched rows require no padding."""
    buckets = defaultdict(list)
    for rec, prompt in zip(records, prompts):
        n = len(processor.tokenizer(prompt, add_special_tokens=True)["input_ids"])
        buckets[n].append((rec, prompt))
    return [buckets[k] for k in sorted(buckets)]


@torch.no_grad()
def generate_batch(model, processor, images, prompts, max_new_tokens, steering=None):
    """Greedy generation for one batch of equal-length prompts; telemetry is cut at each row's EOS."""
    inputs = processor(images=images, text=prompts, return_tensors="pt",
                       padding=True).to(model.device)
    if steering is not None:
        steering.reset()
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         num_beams=1, use_cache=True,
                         pad_token_id=processor.tokenizer.pad_token_id)
    if cuda:
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0

    n_in = inputs["input_ids"].shape[1]
    new = out[:, n_in:].tolist()
    eos = eos_token_ids(model, processor)
    steps = out.shape[1] - n_in
    rows = []
    for i, ids in enumerate(new):
        n = generated_length(ids, eos)
        row = {
            "text": processor.tokenizer.decode(ids[:n], skip_special_tokens=True).strip(),
            "first_token": processor.tokenizer.decode(ids[:1], skip_special_tokens=True).strip(),
            "n_tokens": n,
            "ms_per_token": 1000.0 * seconds / max(1, steps * len(new)),
        }
        hook = steering.hook if steering is not None else None
        if hook is not None:
            row.update(hook.row_telemetry(i, n))
        rows.append(row)
    return rows


def _open_image(record, image_dirs):
    name = record["image"]
    for d in image_dirs:
        path = os.path.join(d, name)
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
    return None


class JsonlStore:
    """Append-only per-sample records keyed by ``uid``, used to resume interrupted runs."""

    def __init__(self, path):
        self.path = path
        self.done = {}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.done[str(r["uid"])] = r

    def add(self, records):
        with open(self.path, "a") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
                self.done[str(r["uid"])] = r

    def ordered(self, records):
        return [self.done[str(r["uid"])] for r in records if str(r["uid"]) in self.done]


def run(model, processor, records, prompt_fn, image_dirs, max_new_tokens,
        batch_size=1, steering=None, store=None, log=print):
    """Generate for records with ``uid`` and ``image``; returns input fields plus outputs, in input order."""
    todo = [r for r in records if store is None or str(r["uid"]) not in store.done]
    results = {}
    buckets = _length_buckets(processor, todo, [prompt_fn(r) for r in todo])
    n_done, t0, next_report = 0, time.time(), max(1, len(todo) // 10)
    for bucket in buckets:
        i, bs = 0, max(1, int(batch_size))
        while i < len(bucket):
            chunk = bucket[i:i + bs]
            kept, images, prompts = [], [], []
            for rec, prompt in chunk:
                img = _open_image(rec, image_dirs)
                if img is None:
                    log(f"missing image, skipped: {rec['image']}")
                    continue
                kept.append(rec)
                images.append(img)
                prompts.append(prompt)
            if not kept:
                i += len(chunk)
                continue
            try:
                rows = generate_batch(model, processor, images, prompts, max_new_tokens, steering)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                log(f"out of memory; batch size reduced to {bs}")
                continue
            out = [{**rec, **row} for rec, row in zip(kept, rows)]
            if store is not None:
                store.add(out)
            for r in out:
                results[str(r["uid"])] = r
            i += len(chunk)
            n_done += len(chunk)
            if n_done >= next_report or n_done == len(todo):
                log(f"{n_done}/{len(todo)} samples, {n_done / max(1e-9, time.time() - t0):.2f}/s")
                next_report += max(1, len(todo) // 10)
    if store is not None:
        return store.ordered(records)
    return [results[str(r["uid"])] for r in records if str(r["uid"]) in results]
