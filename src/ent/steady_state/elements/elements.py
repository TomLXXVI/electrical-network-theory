from __future__ import annotations
from dataclasses import dataclass

from .base import PassiveElement
from .formatting import format_value

__all__ = ["Resistor", "Inductor", "Capacitor"]


@dataclass(frozen=True)
class Resistor(PassiveElement):
    R: float  # ohm

    def Z(self, s: complex) -> complex:
        return complex(self.R)

    def __str__(self) -> str:
        return f"R={format_value(self.R, 'Ω')}"

    def __repr__(self) -> str:
        return f"Resistor(R={self.R})"


@dataclass(frozen=True)
class Inductor(PassiveElement):
    L: float  # henry

    def Z(self, s: complex) -> complex:
        return s * self.L

    def __str__(self) -> str:
        return f"L={format_value(self.L, 'H')}"

    def __repr__(self) -> str:
        return f"Inductor(L={self.L})"


@dataclass(frozen=True)
class Capacitor(PassiveElement):
    C: float  # farad

    def Z(self, s: complex) -> complex:
        if s == 0:
            # In Laplace Z = 1 / (sC) -> infinite at s=0 (DC open circuit)
            return complex("inf")
        return 1 / (s * self.C)

    def __str__(self) -> str:
        return f"C={format_value(self.C, 'F')}"

    def __repr__(self) -> str:
        return f"Capacitor(C={self.C})"
