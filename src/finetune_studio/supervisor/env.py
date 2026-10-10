"""Environment handed to children.

``import finetune_studio`` applies the GPU visibility policy to the importing process
(``accel.env.apply_device_policy``), so the supervisor's own environment carries variables the
policy set from the Settings choice. A child must not inherit those as if an operator had
pinned them: it would report ``source: env`` and ignore a changed Settings choice. Strip exactly
what the policy applied; an operator-set variable (unit file, shell) still passes through.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

from finetune_studio.accel.env import get_applied


def child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    for var, value in get_applied().applied.items():
        if env.get(var) == value:
            del env[var]
    env.update(extra or {})
    return env
