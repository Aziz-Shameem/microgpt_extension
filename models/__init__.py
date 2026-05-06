"""Model registry for different GPT variants."""
from .baseline import gpt_baseline
from .rope import gpt_rope
from .alibi import gpt_alibi
from .t5_bias import gpt_t5_bias
from .flash import gpt_flash
from .xpos import gpt_xpos
from .mtp_naive import gpt_mtp_naive

MODEL_REGISTRY = {
    "baseline": gpt_baseline,
    "rope": gpt_rope,
    "alibi": gpt_alibi,
    "t5_bias": gpt_t5_bias,
    "flash": gpt_flash,
    "xpos": gpt_xpos,
    "mtp_naive": gpt_mtp_naive
}

__all__ = ["MODEL_REGISTRY"]
