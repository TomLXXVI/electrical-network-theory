from ent.steady_state import Resistor, Inductor, Capacitor, Series, j2pif


def main():
    R = Resistor(10.0)
    L = Inductor(50e-3)
    C = Capacitor(100e-6)

    Z_series = Series([R, L, C])

    s = j2pif(50.0)
    z = Z_series.Z(s)
    y = Z_series.Y(s)

    print(f"Z(jw) =", z)
    print(f"|Z| = ", abs(z))
    print(f"Y(jw) = ", y)
    print(f"|Y| = ", abs(y))


if __name__ == "__main__":
    main()
