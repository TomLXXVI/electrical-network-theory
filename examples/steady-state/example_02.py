from ent.steady_state import Resistor, Inductor, Capacitor, j2pif


def main():
    R = Resistor(10.0)
    L = Inductor(50e-3)
    C = Capacitor(100e-6)

    # Series: R + L + C
    Z1 = R + L + C

    # Parallel: (R || C) in series with L
    Z2 = (R | C) + L

    # Repeat in series: 3R
    Z3 = 3 * R

    s = j2pif(50.0)

    print("Z1(jw) =", Z1.Z(s))
    print("Z2(jw) =", Z2.Z(s))
    print("Z3(jw) =", Z3.Z(s))

if __name__ == "__main__":
    main()
