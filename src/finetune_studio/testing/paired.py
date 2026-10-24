"""Is run B really better than run A on the same questions? Exact McNemar test.

Two pass rates on one quiz differ by noise even between identical training runs (Korvane RESULTS §20: two
baseline runs each passed 15 of 102, only 7 in common). Pairing per question removes the questions both runs
pass or both fail; only the discordant ones (gained, lost) carry evidence, and the exact two-sided binomial
test on them says whether the split could be chance (Miller, "Adding Error Bars to Evals", arXiv 2411.00640).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from math import comb
from typing import Any

NOISE_P = 0.05


def mcnemar_exact_p(gained: int, lost: int) -> float:
    """Two-sided exact McNemar p-value (binomial, p = 0.5) for the discordant counts."""
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(gained, lost) + 1)) / 2**n
    return min(1.0, 2 * tail)


@dataclass(frozen=True)
class PairedComparison:
    """Per-question comparison of run B against baseline A on the cases both answered."""

    paired: int
    both_pass: int
    gained: int  # B passes, A fails
    lost: int  # A passes, B fails
    both_fail: int
    p_value: float
    verdict: str  # "better" | "worse" | "could be noise" | "no shared questions"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_outcomes(a: Mapping[str, bool], b: Mapping[str, bool]) -> PairedComparison:
    """Pair two ``case key -> passed`` maps on their shared keys."""
    keys = sorted(set(a) & set(b))
    both = sum(1 for k in keys if a[k] and b[k])
    gained = sum(1 for k in keys if b[k] and not a[k])
    lost = sum(1 for k in keys if a[k] and not b[k])
    p = mcnemar_exact_p(gained, lost)
    if not keys:
        verdict = "no shared questions"
    elif p >= NOISE_P:
        verdict = "could be noise"
    else:
        verdict = "better" if gained > lost else "worse"
    return PairedComparison(len(keys), both, gained, lost, len(keys) - both - gained - lost, p, verdict)


def case_outcomes(cases: Iterable[Mapping[str, Any]]) -> dict[str, bool]:
    """``case key -> passed`` for the scored cases of one benchmark run (unscored cases are left out).

    The key is case name + question, so the same suite (or the same seeded sample of an official one) pairs
    across runs and two cases sharing a name stay apart. Only ``pass`` counts as passed.
    """
    out: dict[str, bool] = {}
    for case in cases:
        verdict = str(case.get("verdict") or "")
        if not verdict:
            continue
        key = f"{str(case.get('case_name') or '').strip()}\x1f{str(case.get('question') or '').strip()}"
        if key != "\x1f":
            out[key] = verdict == "pass"
    return out
