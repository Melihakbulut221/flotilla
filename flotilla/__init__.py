"""flotilla: self-hosted sandbox swarms and RL environments for LLM training."""

__version__ = "0.1.0"

from .spec import SandboxInput, Size, SpecError, SwarmSpec  # noqa: E402
from .client import Flotilla  # noqa: E402

__all__ = ["Flotilla", "SwarmSpec", "SandboxInput", "Size", "SpecError", "__version__"]
