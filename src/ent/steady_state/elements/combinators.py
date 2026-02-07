from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Iterable
from .base import PassiveElement
from .elements import Resistor

__all__ = ["Series", "Parallel"]


def _flatten_series(items: Iterable[PassiveElement]) -> list[PassiveElement]:
    flat: list[PassiveElement] = []
    for it in items:
        if isinstance(it, Series):
            flat.extend(_flatten_series(it.elements))
        else:
            flat.append(it)
    return flat


def _flatten_parallel(items: Iterable[PassiveElement]) -> list[PassiveElement]:
    flat: list[PassiveElement] = []
    for it in items:
        if isinstance(it, Parallel):
            flat.extend(_flatten_parallel(it.elements))
        else:
            flat.append(it)
    return flat


def _is_inf(z: complex) -> bool:
    return z.real == float("inf") or z.imag == float("inf")


@dataclass(frozen=True)
class Series(PassiveElement):
    elements: Sequence[PassiveElement]

    @classmethod
    def make(cls, items: Sequence[PassiveElement]) -> PassiveElement:
        """Factory that flattens and applies basic simplifications."""
        flat = _flatten_series(items)

        # Remove 0 ohm resistors (shorts) in series (no effect)
        flat2: list[PassiveElement] = []
        for it in flat:
            if isinstance(it, Resistor) and it.R == 0:
                continue
            flat2.append(it)

        # Combine resistors in series
        r_sum = 0.0
        rest: list[PassiveElement] = []
        for it in flat2:
            if isinstance(it, Resistor):
                r_sum += it.R
            else:
                rest.append(it)

        out: list[PassiveElement] = []
        if r_sum != 0.0:
            out.append(Resistor(r_sum))
        out.extend(rest)

        if len(out) == 0:
            return Resistor(0.0)
        if len(out) == 1:
            return out[0]
        return cls(out)

    def Z(self, s: complex) -> complex:
        return sum(elem.Z(s) for elem in self.elements)

    def __str__(self) -> str:
        if hasattr(self, "_repeat"):
            # noinspection PyUnresolvedReferences
            return f"{self._repeat}·{self._base}"
        inner = " + ".join(str(it) for it in self.elements)
        return f"({inner})"

    def __repr__(self) -> str:
        return f"Series({self.elements!r})"


@dataclass(frozen=True)
class Parallel(PassiveElement):
    elements: Sequence[PassiveElement]

    @classmethod
    def make(cls, items: Sequence[PassiveElement]) -> PassiveElement:
        """Factory that flattens and applies basic simplifications."""
        flat = _flatten_parallel(items)

        # Remove infinite impedances in parallel (open branches have no effect)
        flat2: list[PassiveElement] = []
        for it in flat:
            # We judge infinity by checking Z at s=1 (arbitrary nonzero)
            z1 = it.Z(1.0 + 0j)
            if _is_inf(z1):
                continue
            flat2.append(it)

        # Combine resistors in parallel: 1/R_eq = sum(1/Rk)
        inv_r_sum = 0.0
        rest: list[PassiveElement] = []
        for it in flat2:
            if isinstance(it, Resistor):
                if it.R == 0.0:
                    return Resistor(0.0)  # short in parallel dominates
                inv_r_sum += 1.0 / it.R
            else:
                rest.append(it)

        out: list[PassiveElement] = []
        if inv_r_sum != 0.0:
            out.append(Resistor(1.0 / inv_r_sum))
        out.extend(rest)

        if len(out) == 0:
            return Resistor(float("inf"))  # empty parallel = open
        if len(out) == 1:
            return out[0]
        return cls(out)

    def Z(self, s: complex) -> complex:
        y_total = 0j
        for elem in self.elements:
            y_total += elem.Y(s)
        if y_total == 0:
            return complex("inf")
        return 1 / y_total

    def __str__(self) -> str:
        inner = " || ".join(str(it) for it in self.elements)
        return f"({inner})"

    def __repr__(self) -> str:
        return f"Parallel({self.elements!r})"
