__all__ = ["delta_to_wye", "wye_to_delta"]


def delta_to_wye(
    Z_a: complex,
    Z_b: complex | None = None,
    Z_c: complex | None = None
) -> tuple[complex, complex, complex]:
    """
    Delta-to-star transformation.

    Parameters
    ----------
    Z_a: complex
        Impedance of phase leg 'a'.
    Z_b: complex, optional
        Impedance of phase leg 'b'. If None, Z_b is set equal to Z_a.
    Z_c: complex, optional
        Impedance of phase leg 'c'. If None, Z_c is set equal to Z_a.

    Returns
    -------
    tuple[complex, complex, complex]
        Z_alpha:
            Equivalent impedance of phase leg 'a' when delta connection would
            be replaced by wye connection.
        Z_beta:
            Equivalent impedance of phase leg 'b' when delta connection would
            be replaced by wye connection.
        Z_gamma:
            Equivalent impedance of phase leg 'c' when delta connection would
            be replaced by wye connection.
    """
    if Z_b is None:
        Z_b = Z_a
    if Z_c is None:
        Z_c = Z_a
    den = Z_a + Z_b + Z_c
    Z_alpha = Z_b * Z_c / den
    Z_beta = Z_a * Z_c / den
    Z_gamma = Z_a * Z_b / den
    return Z_alpha, Z_beta, Z_gamma


def wye_to_delta(
    Z_alpha: complex,
    Z_beta: complex | None = None,
    Z_gamma: complex | None = None
) -> tuple[complex, complex, complex]:
    """
    Star-to-delta transformation.

    Parameters
    ----------
    Z_alpha: complex
        Impedance of phase leg 'a'.
    Z_beta: complex, optional
        Impedance of phase leg 'b'. If None, Z_beta is set equal to Z_alpha.
    Z_gamma: complex, optional
        Impedance of phase leg 'c'. If None, Z_gamma is set equal to Z_alpha.

    Returns
    -------
    tuple[complex, complex, complex]
        Z_a:
            Equivalent impedance of phase leg 'a' when delta connection would
            be replaced by wye connection.
        Z_b:
            Equivalent impedance of phase leg 'b' when delta connection would
            be replaced by wye connection.
        Z_c:
            Equivalent impedance of phase leg 'c' when delta connection would
            be replaced by wye connection.
    """
    if Z_beta is None:
        Z_beta = Z_alpha
    if Z_gamma is None:
        Z_gamma = Z_alpha
    num = Z_alpha * Z_beta + Z_beta * Z_gamma + Z_alpha * Z_gamma
    Z_a = num / Z_alpha
    Z_b = num / Z_beta
    Z_c = num / Z_gamma
    return Z_a, Z_b, Z_c
