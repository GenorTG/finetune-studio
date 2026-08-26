from typing import Any

"""Sampler configuration presets.

WHAT THIS FILE DOES
==================
Defines named sampler configurations (temperature, top_p, etc.) for
different use cases:
  - "creative": high temperature (0.9), high top_p (0.95) for stories
  - "balanced": medium temperature (0.7), default top_p (0.9) for chat
  - "precise": low temperature (0.3), low top_p (0.8) for factual Q&A
  - "greedy": temperature=0 for deterministic answers

KEY CONCEPTS
============
- Sampler parameters: see samplers.py in inference-server for full details.
- Preset names: a friendly way to refer to a configuration without
  remembering all the numbers.
- Custom presets: users can define their own presets in config files.
"""

"""Sampler options for inference — industry-standard parameters."""
from dataclasses import dataclass, field


@dataclass
class SamplerConfig:
    """All standard sampler parameters for LLM inference."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    min_p: float = 0.05
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 512
    stop: list = field(default_factory=list)
    seed: int = -1  # -1 = random

    def to_llama_cpp_kwargs(self) -> dict:
        """Convert to llama-cpp-python kwargs."""
        kwargs: dict[str, Any] = {
            "max_tokens": self.max_tokens,
            "temperature": max(self.temperature, 0.01),
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "min_p": self.min_p,
        }
        if self.seed >= 0:
            kwargs["seed"] = self.seed
        if self.stop:
            kwargs["stop"] = self.stop
        return kwargs

    def to_openai_kwargs(self) -> dict:
        """Convert to OpenAI API kwargs."""
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SamplerConfig":
        """Create from dict, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# Preset sampler configurations
PRESETS = {
    "deterministic": SamplerConfig(temperature=0.0, top_p=1.0, top_k=1, repeat_penalty=1.0, min_p=0.0),
    "balanced": SamplerConfig(temperature=0.7, top_p=0.9, top_k=40, repeat_penalty=1.1, min_p=0.05),
    "creative": SamplerConfig(temperature=1.0, top_p=0.95, top_k=100, repeat_penalty=1.2, min_p=0.02),
    "conservative": SamplerConfig(temperature=0.3, top_p=0.85, top_k=20, repeat_penalty=1.05, min_p=0.08),
    "chris_ai_v20": SamplerConfig(temperature=0.45, top_p=0.9, top_k=30, repeat_penalty=1.02, min_p=0.02),
    "chris_ai_v21": SamplerConfig(temperature=0.25, top_p=0.8, top_k=15, repeat_penalty=1.05, min_p=0.03),
    "testing": SamplerConfig(temperature=0.0, top_p=1.0, top_k=1, repeat_penalty=1.0, min_p=0.0, max_tokens=256),
}


def get_sampler(name: str = "balanced") -> SamplerConfig:
    """Get a preset sampler by name."""
    return PRESETS.get(name, PRESETS["balanced"])


def list_presets() -> dict:
    """List all preset samplers."""
    return {name: {
        "temperature": p.temperature,
        "top_p": p.top_p,
        "top_k": p.top_k,
        "repeat_penalty": p.repeat_penalty,
        "min_p": p.min_p,
    } for name, p in PRESETS.items()}
