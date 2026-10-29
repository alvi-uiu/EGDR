"""Model loading, module access and prompt construction for LLaVA backbones."""

import torch


def load_model(hf_id: str, dtype: str = "float16", attn_implementation: str = "eager",
               device: str = "cuda:0"):
    """Load a frozen LLaVA-1.5 or LLaVA-NeXT checkpoint and its processor."""
    from transformers import AutoProcessor
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()

    lowered = hf_id.lower()
    if "v1.6" in lowered or "next" in lowered:
        from transformers import LlavaNextForConditionalGeneration as ModelClass
    else:
        from transformers import LlavaForConditionalGeneration as ModelClass

    processor = AutoProcessor.from_pretrained(hf_id)
    model = ModelClass.from_pretrained(
        hf_id,
        torch_dtype=getattr(torch, dtype),
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # expand image tokens in the processor so batched and single prompts share one layout
    if getattr(processor, "patch_size", None) is None:
        processor.patch_size = model.config.vision_config.patch_size
    if getattr(processor, "vision_feature_select_strategy", None) is None:
        processor.vision_feature_select_strategy = getattr(
            model.config, "vision_feature_select_strategy", "default")
    return model, processor


def decoder_layers(model):
    """Decoder blocks of the language model (transformers 4.46 and >= 4.52 layouts)."""
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    if lm is not None and hasattr(lm, "layers"):
        return lm.layers
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "language_model"):
        return inner.language_model.layers
    raise ValueError(f"cannot locate decoder layers of {type(model).__name__}")


def lm_head(model):
    """Output projection producing next-token logits."""
    lm = getattr(model, "language_model", None)
    if lm is not None and hasattr(lm, "lm_head"):
        return lm.lm_head
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise ValueError(f"cannot locate lm_head of {type(model).__name__}")


def build_prompt(processor, text: str) -> str:
    """Single-turn chat prompt with one image followed by ``text``."""
    conversation = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": text}],
    }]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def build_text_prompt(processor, text: str) -> str:
    """Single-turn chat prompt without visual tokens."""
    conversation = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def eos_token_ids(model, processor):
    """Token ids that terminate a generation."""
    ids = getattr(model.generation_config, "eos_token_id", None)
    if ids is None:
        ids = processor.tokenizer.eos_token_id
    if isinstance(ids, int):
        ids = [ids]
    return set(int(i) for i in ids)
