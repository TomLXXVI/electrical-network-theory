from __future__ import annotations
from dataclasses import dataclass
from .base import PassiveElement


@dataclass(frozen=True)
class RepeatSeries(PassiveElement):
    n: int
    base: PassiveElement

    def __post_init__(self) -> None:
        if self.n < 0:
            raise ValueError("n must be >= 0")

    def Z(self, s: complex) -> complex:
        return self.n * self.base.Z(s)

    def __str__(self) -> str:
        # Parentheses only if base is a composite
        from .combinators import Series, Parallel  # safe local import
        if isinstance(self.base, (Series, Parallel)):
            return f"{self.n}·{self.base}"
        return f"{self.n}·{self.base}"

    def __repr__(self) -> str:
        return f"RepeatSeries(n={self.n}, base={self.base!r})"
