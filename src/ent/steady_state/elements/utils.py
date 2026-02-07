from __future__ import annotations
import math

__all__ = ["j2pif"]


def jw(omega: float) -> complex:
    return 1j * omega


def j2pif(f_hz: float) -> complex:
    return 1j * 2 * math.pi * f_hz
