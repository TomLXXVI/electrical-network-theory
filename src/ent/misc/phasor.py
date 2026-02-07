from dataclasses import dataclass
import math

__all__ = ["Phasor"]


@dataclass(frozen=True)
class Phasor:
    value: complex

    @property
    def magnitude(self) -> float:
        return abs(self.value)

    @property
    def mag(self) -> float:
        return self.magnitude

    @property
    def angle_rad(self) -> float:
        return math.atan2(self.value.imag, self.value.real)

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle_rad)

    def __str__(self) -> str:
        return f"|{self.mag:.6g}| < {self.angle_deg:.6g}°"
