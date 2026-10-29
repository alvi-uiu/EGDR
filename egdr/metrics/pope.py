"""POPE metrics with "yes" as the positive class, at answer level and at the first token."""

import re

_SENTENCE_END = re.compile(r"[.!?\n]")
_WORD = re.compile(r"[a-z']+")


def parse_yes_no(text):
    """Answer-level label (Sec. 5.1): the first sentence is ``"no"`` if it contains a standalone
    "no" or "not" and ``"yes"`` otherwise; an empty answer is ``"unk"`` and counts as "no"."""
    t = (text or "").strip().lower()
    if not t:
        return "unk"
    words = _WORD.findall(_SENTENCE_END.split(t, 1)[0])
    return "no" if "no" in words or "not" in words else "yes"


def first_token_label(token):
    """First-token label (Sec. 3.4, Tab. S6): the decoded first token read as yes, no or unk."""
    t = (token or "").strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return "unk"


def pope_metrics(records, parse=parse_yes_no, field="text"):
    """Accuracy, precision, recall and F1 in percent; an ``"unk"`` label counts as "no" (``unk_rate``)."""
    tp = fp = tn = fn = unk = 0
    for r in records:
        pred = parse(r.get(field))
        gt = str(r.get("label", "")).lower()
        if pred == "unk":
            unk += 1
            pred = "no"
        if gt == "yes" and pred == "yes":
            tp += 1
        elif gt == "no" and pred == "yes":
            fp += 1
        elif gt == "no" and pred == "no":
            tn += 1
        elif gt == "yes" and pred == "no":
            fn += 1
    n = max(1, tp + fp + tn + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "acc": 100.0 * (tp + tn) / n,
        "precision": 100.0 * precision,
        "recall": 100.0 * recall,
        "f1": 100.0 * f1,
        "yes_rate": 100.0 * (tp + fp) / n,
        "unk_rate": 100.0 * unk / n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": tp + fp + tn + fn,
    }


def first_token_metrics(records):
    """``pope_metrics`` of the first generated token (``record["first_token"]``)."""
    return pope_metrics(records, parse=first_token_label, field="first_token")
