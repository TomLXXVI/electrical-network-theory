"""
RLC series circuit.
"""

import matplotlib.pyplot as plt

from ent.transient import Network, simulate_network, sine_waveform_rms


def main() -> None:
    # --------------------------------------------------------------------------
    # Build network
    # --------------------------------------------------------------------------
    nw = Network()
    nw.add_node("0")  # -> network reference node

    # Voltage source: step to 10 V at t = 0.
    nw.add_voltage_source(
        "V1", "A", "0",
        v=sine_waveform_rms(220, 50))

    # Series R-L between A and B
    nw.add_resistor("R1", "A", "X", R=0.1)
    nw.add_inductor("L1", "X", "B", L=200e-6, i0=0.0)

    # Capacitor bank to ground
    nw.add_capacitor("C1", "B", "0", C=2e-3, v0=0.0)

    # --------------------------------------------------------------------------
    # Solve network
    # --------------------------------------------------------------------------
    sol = simulate_network(nw, t_end=0.05, dt=1e-6)
    print(sol)

    # --------------------------------------------------------------------------
    # Study results
    # --------------------------------------------------------------------------
    vB = sol.node_voltage("B")
    iL = sol.inductor_current("L1")

    plt.figure()
    plt.plot(sol.t, vB)
    plt.xlabel("t [s]")
    plt.ylabel("v_B(t) [V]")
    plt.grid(True)

    plt.figure()
    plt.plot(sol.t, iL)
    plt.xlabel("t [s]")
    plt.ylabel("i_L(t) [A]  (positive A→B)")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
