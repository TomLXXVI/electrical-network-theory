from __future__ import annotations


def format_value(value: float, unit: str) -> str:
    prefixes = [
        (1e9, "G"),
        (1e6, "M"),
        (1e3, "k"),
        (1.0, ""),
        (1e-3, "m"),
        (1e-6, "µ"),
        (1e-9, "n"),
        (1e-12, "p"),
    ]

    for scale, prefix in prefixes:
        if abs(value) >= scale:
            return f"{value / scale:g}{prefix}{unit}"
    return f"{value:g}{unit}"
