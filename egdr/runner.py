"""Running one arm over one set of records, with resumable output and gate statistics."""

import hashlib
import json
import os
import time

import numpy as np

from . import generation
from .hooks import Steering


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fingerprint(**fields):
    """Hash of the run settings; output files are keyed by it so that runs never mix settings."""
    blob = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def run_arm(model, processor, records, prompt_fn, image_dirs, max_new_tokens, batch_size,
            spec, direction, out_path):
    """Generate for ``records`` under one arm; returns the output records."""
    if direction is not None:
        digest = hashlib.sha256(direction.vector.numpy().tobytes()).hexdigest()[:8]
        stem, ext = os.path.splitext(out_path)
        out_path = f"{stem}-v{digest}{ext}"
    store = generation.JsonlStore(out_path)
    pending = sum(str(r["uid"]) not in store.done for r in records)
    log(f"{os.path.basename(os.path.dirname(out_path))}/{os.path.basename(out_path)}: "
        f"{len(records) - pending} cached, {pending} to run")
    with Steering(model, direction, spec) as st:
        out = generation.run(model, processor, records, prompt_fn, image_dirs,
                             max_new_tokens, batch_size=batch_size,
                             steering=st if spec is not None else None,
                             store=store, log=log)
        if len(out) < len(records):
            log(f"warning: {len(records) - len(out)} of {len(records)} records produced no output")
        if st.tracker is not None and st.tracker.n_nonfinite:
            log(f"non-finite logits guarded in {st.tracker.n_nonfinite} rows")
    return out


def gate_statistics(records):
    """Percentage of decoding steps (up to each sample's EOS) at which the gate opens, and ms per token."""
    fired = [f for r in records for f in r.get("fired", [])]
    ms = [r["ms_per_token"] for r in records if "ms_per_token" in r]
    return {
        "fire_rate": 100.0 * float(np.mean(fired)) if fired else None,
        "n_decoding_steps": len(fired),
        "ms_per_token": float(np.mean(ms)) if ms else None,
    }
