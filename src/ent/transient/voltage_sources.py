from __future__ import annotations
import math
from typing import Callable


__all__ = [
    "step_waveform",
    "sine_waveform",
    "sine_waveform_rms",
    "damped_sine_waveform"
]


def step_waveform(V: float, t_start: float = 0.0) -> Callable[[float], float]:
    """
    Returns a voltage step-function.

    Parameters
    ----------
    V: float
        Constant voltage value in Volts.
    t_start: float
        Time moment in seconds from t = 0 where the voltage steps up from 0 to V.
    """
    return lambda t: V if t >= t_start else 0.0


def sine_waveform(
    amplitude: float,
    frequency_hz: float,
    *,
    phase_rad: float = 0.0,
    offset: float = 0.0,
    t_start: float = 0.0,
) -> Callable[[float], float]:
    """
    Returns a time-dependent sinusoidal waveform v(t).

    For t < t_start:
        offset

    For t >= t_start:
        v(t) = offset + amplitude * sin(2*pi*frequency_hz*(t - t_start) + phase_rad)

    Parameters
    ----------
    amplitude: float
        Peak amplitude (not RMS).
    frequency_hz: float
        Frequency in Hz.
    phase_rad: float
        Phase shift in radians.
    offset: float
        DC offset added to the sine.
    t_start: float
        Start time in seconds. Before this time the waveform equals `offset`.
    """
    if frequency_hz < 0:
        raise ValueError("frequency_hz must be >= 0")
    if t_start < 0:
        raise ValueError("t_start must be >= 0")

    w = 2.0 * math.pi * frequency_hz

    def v(t: float) -> float:
        if t < t_start:
            return offset
        return offset + amplitude * math.sin(w * (t - t_start) + phase_rad)

    return v


def sine_waveform_rms(
    vrms: float,
    frequency_hz: float,
    *,
    phase_rad: float = 0.0,
    offset: float = 0.0,
    t_start: float = 0.0,
) -> Callable[[float], float]:
    """
    Same as `sine_waveform`, but the amplitude is specified as RMS.

    amplitude = vrms * sqrt(2)
    """
    return sine_waveform(
        amplitude=vrms * math.sqrt(2.0),
        frequency_hz=frequency_hz,
        phase_rad=phase_rad,
        offset=offset,
        t_start=t_start,
    )


def damped_sine_waveform(
    amplitude: float,
    frequency_hz: float,
    *,
    damping_per_s: float = 0.0,
    phase_rad: float = 0.0,
    offset: float = 0.0,
    t_start: float = 0.0,
) -> Callable[[float], float]:
    """
    Damped sinusoidal waveform v(t) with exponential envelope.

    For t >= t_start:
        v(t) = offset + amplitude * exp(-damping_per_s*(t - t_start)) * sin(2*pi*f*(t - t_start) + phase)

    Parameters
    ----------
    amplitude: float
        Peak amplitude (not RMS).
    frequency_hz: float
        Frequency in Hz.
    damping_per_s:
        Exponential damping coefficient [1/s]. Use 0 for an undamped sine.
    phase_rad: float
        Phase shift in radians.
    offset: float
        DC offset added to the sine.
    t_start: float
        Start time in seconds. Before this time the waveform equals `offset`.
    """
    if damping_per_s < 0:
        raise ValueError("damping_per_s must be >= 0")

    w = 2.0 * math.pi * frequency_hz

    def v(t: float) -> float:
        if t < t_start:
            return offset
        tau = t - t_start
        return offset + amplitude * math.exp(-damping_per_s * tau) * math.sin(w * tau + phase_rad)

    return v
