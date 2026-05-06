"""Model registry for different GPT variants."""
from .baseline import gpt_baseline
from .rope import gpt_rope
from .alibi import gpt_alibi
from .t5_bias import gpt_t5_bias
from .flash import gpt_flash
from .xpos import gpt_xpos

MODEL_REGISTRY = {
    "baseline": gpt_baseline,
    "rope": gpt_rope,
    "alibi": gpt_alibi,
    "t5_bias": gpt_t5_bias,
    "flash": gpt_flash,
    "xpos": gpt_xpos,
}

__all__ = ["MODEL_REGISTRY"]
