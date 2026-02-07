from ent.steady_state import Network, Resistor, Inductor, Capacitor, j2pif


def main():
    nw = Network()

    nw.add_passive_element("R1", "A", "C", Resistor(10.0))
    nw.add_passive_element("L1", "C", "B", Inductor(50e-3))
    nw.add_passive_element("C1", "C", "B", Capacitor(100e-6))
    nw.add_passive_element("R2", "A", "B", Resistor(100.0))

    print(nw)

    s = j2pif(f_hz=50.0)

    z_eq = nw.equivalent_impedance("A", "B", s)
    print("\nZ_eq(A, B)", z_eq)
    print("|Z_eq| =", abs(z_eq))


if __name__ == '__main__':
    main()
