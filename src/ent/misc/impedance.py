from __future__ import annotations
import cmath
import math

from dataclasses import dataclass, field

__all__ = ["Impedance", "ImpedancePower", "pow_to_imp"]


@dataclass
class Impedance:
    """
    Wrapper class around a complex number that represents the impedance of
    a passive network element.

    Attributes
    ----------
    value: complex
        The complex number that represents an impedance.
    pow: ImpedancePower
        Internal object of Impedance (not in __init__()-method). It is
        instantiated when either the voltage across or the current through the
        impedance is set (see method set_voltage() or set_current()). After
        either the voltage or current has been set, the power of the impedance
        (apparent, reactive, active) can be retrieved through attribute pow.
        See class ImpedancePower.
    """
    value: complex
    pow: ImpedancePower = field(init=False, default=None)

    @property
    def R(self) -> float:
        return self.value.real

    @property
    def X(self) -> float:
        return self.value.imag

    @property
    def magnitude(self) -> float:
        return abs(self.value)

    @property
    def mag(self) -> float:
        return self.magnitude

    @property
    def angle_rad(self) -> float:
        return cmath.phase(self.value)

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle_rad)

    @property
    def is_inductive(self) -> bool:
        return self.angle_rad > 0.0

    @property
    def is_capacitive(self) -> bool:
        return self.angle_rad < 0.0

    @property
    def is_resistive(self) -> bool:
        return self.angle_rad == 0.0

    @property
    def phi_rad(self) -> float:
        return self.angle_rad

    @property
    def phi_deg(self) -> float:
        return self.angle_deg

    @property
    def cos_phi(self) -> float:
        return math.cos(self.phi_rad)

    @property
    def sin_phi(self) -> float:
        return math.sin(self.phi_rad)

    def set_voltage(self, U: float) -> None:
        """Sets the voltage across the impedance."""
        self.pow = ImpedancePower(self, U)

    def set_current(self, I: float) -> None:
        """Sets the current through the impedance."""
        self.pow = ImpedancePower(self, I)


@dataclass
class ImpedancePower:
    """
    Functional class to determine the power (apparent, active, and reactive) of
    an impedance. To determine the power, either the voltage across the
    impedance or the current through the impedance must be known.

    Attributes
    ----------
    Z: Impedance
        The impedance in question.
    U: float, optional
        Voltage across the impedance.
    I: float, optional
        Current through the impedance.
    """
    Z: Impedance
    U: float | None = None
    I: float | None = None

    _UI: float = field(init=False, default=None)

    def __post_init__(self):
        if self.U is not None:
            self._UI = (self.U**2 / self.Z.mag)
        elif self.I is not None:
            self._UI = self.Z.mag * self.I**2
        else:
            raise ValueError("Either 'U' or 'I' must be specified.")

    @property
    def cos_phi(self) -> float:
        return self.Z.cos_phi

    @property
    def sin_phi(self) -> float:
        return self.Z.sin_phi

    @property
    def apparent_power(self) -> complex:
        P = self._UI * self.cos_phi
        Q = self._UI * self.sin_phi
        S = P + 1j * Q
        return S

    @property
    def active_power(self) -> float:
        return self.apparent_power.real

    @property
    def reactive_power(self) -> float:
        return self.apparent_power.imag

    @property
    def S(self) -> complex:
        return self.apparent_power

    @property
    def P(self) -> float:
        return self.active_power

    @property
    def Q(self) -> float:
        return self.reactive_power


def pow_to_imp(
    phi_deg: float,
    U: float | None = None,
    I: float | None = None,
    P: float | None = None,
    Q: float | None = None,
    S: float | None = None
) -> complex:
    """
    Retrieves the complex value of the impedance of a passive network element
    from power data.

    Parameters
    ----------
    phi_deg: float
        The phase shift angle in degrees between the voltage across the
        impedance and the current through the impedance. This parameter must
        always be specified. It can be obtained from the power factor cos_phi.
    U: float, optional
        Voltage across the impedance. If None, current must be specified.
    I: float, optional
        Current through impedance. If None, voltage must be specified.
    P: float, optional
        Active power dissipated as heat by the impedance. If None, either Q or
        S must be specified.
    Q: float, optional
        Reactive power of the impedance. If None, either P or S must be
        specified.
    S: float, optional
        Apparent power of the impedance. If None, either P or Q must be
        specified.

    Returns
    -------
    complex
    """
    if not (
        isinstance(phi_deg, float)
        and (0.0 <= phi_deg <= 180.0)
        or (-180.0 <= phi_deg < 0.0)
    ):
        raise ValueError(
            "Parameter 'phi_deg' is not a valid angle. "
            "Must be between 0° and 360° (not included)."
        )
    phi_rad = math.radians(phi_deg)
    if S is not None:
        if U is not None:
            mag = U**2 / S
            val = cmath.rect(mag, phi_rad)
            return val
        if I is not None:
            mag = S / I**2
            val = cmath.rect(mag, phi_rad)
            return val
    if P is not None:
        cos_phi = math.cos(phi_rad)
        if U is not None:
            mag = U**2 * cos_phi / P
            val = cmath.rect(mag, phi_rad)
            return val
        if I is not None:
            mag = P / (I**2 * cos_phi)
            val = cmath.rect(mag, phi_rad)
            return val
    if Q is not None:
        sin_phi = math.sin(phi_rad)
        if U is not None:
            mag = U**2 * sin_phi / Q
            val = cmath.rect(mag, phi_rad)
            return val
        if I is not None:
            mag = Q / (I**2 * sin_phi)
            val = cmath.rect(mag, phi_rad)
            return val
    raise AttributeError("Parameters seem to be missing.")
