from __future__ import annotations
from abc import ABC, abstractmethod
import math


__all__ = ["PassiveElement"]


class PassiveElement(ABC):
    """
    Abstract base for passive network elements in the Laplace domain.

    Z(s) returns complex impedance for complex frequency s.
    """
    @abstractmethod
    def Z(self, s: complex) -> complex:
        raise NotImplementedError

    def Y(self, s: complex) -> complex:
        """Admittance Y(s) = 1 / Z(s)."""
        z = self.Z(s)
        if z == 0:
            raise ZeroDivisionError("Admittance is infinite because Z(s)=0.")
        return 1 / z

    def Z_jw(self, omega: float) -> complex:
        return self.Z(1j * omega)

    def Y_jw(self, omega: float) -> complex:
        return self.Y(1j * omega)

    def Z_f(self, f_hz: float) -> complex:
        return self.Z(1j * 2 * math.pi * f_hz)

    def Y_f(self, f_hz: float) -> complex:
        return self.Y(1j * 2 * math.pi * f_hz)

    def __add__(self, other: PassiveElement) -> PassiveElement:
        from .combinators import Series
        if not isinstance(other, PassiveElement):
            return NotImplemented
        return Series.make([self, other])

    def __radd__(self, other: PassiveElement) -> PassiveElement:
        if other == 0:
            return self
        if not isinstance(other, PassiveElement):
            return NotImplemented
        return other.__add__(self)

    def __or__(self, other: PassiveElement) -> PassiveElement:
        from .combinators import Parallel
        if not isinstance(other, PassiveElement):
            return NotImplemented
        return Parallel.make([self, other])

    def __ror__(self, other: PassiveElement) -> PassiveElement:
        if not isinstance(other, PassiveElement):
            return NotImplemented
        return other.__or__(self)

    def __mul__(self, n: int) -> PassiveElement:
        """Repeat in series: n * Z = Z + Z + ... + Z (n times)."""
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            raise ValueError("n must be >= 0")

        from .elements import Resistor
        from .repeat import RepeatSeries
        if n == 0:
            return Resistor(0.0)
        if n == 1:
            return self
        return RepeatSeries(n=n, base=self)

    def __rmul__(self, n: int) -> PassiveElement:
        return self.__mul__(n)
