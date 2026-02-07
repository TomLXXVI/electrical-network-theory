"""
Investigation of the influence of harmonic currents on a capacitor bank in an
electrical LV-network. The LV-network is fed from the medium-voltage network via
a transformer.

The harmonic current is modeled by a harmonic current source that injects a
current of 1 A with a certain frequency into the circuit formed by the
transformer and the capacitor bank. In an equivalent single-phase circuit, this
circuit is represented by a parallel connection of an ideal capacitor with a
real inductor (series connection of a resistor and an inductor).

The relationship between the circuit's frequency-dependent impedance and the
frequency of the harmonic current is investigated. Furthermore, the relationship
between the magnitude and the frequency of the harmonic current flowing through
the capacitor is investigated.

Parallel resonance occurs at the frequency of the harmonic current at which the
impedance of the parallel circuit becomes maximum. The value of this frequency
will depend on the number of capacitor bank stages that are switched on.
"""
import math
from dataclasses import dataclass, field

from ent.steady_state import Resistor, Inductor, Capacitor, j2pif, Network, PassiveElement
from ent.misc import Impedance, Phasor, pow_to_imp, delta_to_wye


@dataclass
class CapacitorBank:
    """
    Represents a capacitor bank composed of 4 switchable stages. Each stage
    is delta connected. The single-phase reactive power of each stage is known
    at rated voltage and rated angular frequency (rad/s). Six different
    switching combinations are considered:
    - stage 1 ON
    - stage 1 and 2 ON
    - stage 1 and 3 ON
    - stage 1, 2, and 3 ON
    - stage 1, 3, and 4 ON
    - stage 1, 2, 3, and 4 ON
    """
    Q_stage1: float = 11.845e3 / 3  # reactive power per phase
    Q_stage2: float = 11.845e3 / 3
    Q_stage3: float = 23.689e3 / 3
    Q_stage4: float = 23.689e3 / 3
    V_rated: float = 404.0  # rated voltage across one leg of the capacitor bank.
    omega: float = 2 * math.pi * 50.0

    combinations: dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.combinations = {
            "stage 1": self.Q_stage1,
            "stage 1+2": self.Q_stage1 + self.Q_stage2,
            "stage 1+3": self.Q_stage1 + self.Q_stage3,
            "stage 1+2+3": self.Q_stage1 + self.Q_stage2 + self.Q_stage3,
            "stage 1+3+4": self.Q_stage1 + self.Q_stage3 + self.Q_stage4,
            "stage 1+2+3+4": self.Q_stage1 + self.Q_stage2 + self.Q_stage3 + self.Q_stage4
        }

    def get_C(self, active_combination: str) -> float:
        """
        Determines the capacitor value C from the reactive power Qc.
        """
        # Get the reactive power of the active combination.
        Qc = self.combinations[active_combination]
        # The capacitor impedance value can be derived from the reactive power.
        Zcd_val = pow_to_imp(
            phi_deg=-90.0,
            U=self.V_rated,
            Q=Qc
        )
        # The legs of the capacitor bank are connected in delta. Zcd_val is the
        # impedance value of a single leg in the delta connection.
        # We'll work with an equivalent single-phase network, so we want to
        # replace the delta connection by an equivalent wye connection.
        Zcy_val, *_ = delta_to_wye(Zcd_val)
        C = 1 / (self.omega * abs(Zcy_val))
        return C

    def get_C_current(
        self,
        sol: Solution,
        active_combination: str
    ) -> float:
        """
        Determines the current through a single capacitor given the harmonic
        line current that flows into the capacitor bank.

        Parameters
        ----------
        sol: Solution
            The solution of the network calculation (see method solve() of class
            Equiv1PhNetwork).
        active_combination: str
            Specifies which combination of capacitor bank stages is switched on.
        """
        I_line = abs(sol.I_C.value)  # harmonic line current that flows into the capacitor bank

        combinations: list[str] = list(self.combinations.keys())

        # Each active combination corresponds to a number of identical
        # capacitors connected in parallel.
        if active_combination == combinations[0]:
            n = 1
        elif active_combination == combinations[1]:
            n = 2
        elif active_combination == combinations[2]:
            n = 3
        elif active_combination == combinations[3]:
            n = 4
        elif active_combination == combinations[4]:
            n = 5
        elif active_combination == combinations[5]:
            n = 6
        else:
            raise ValueError("Unkown active combination.")

        # Capacitor current in the equivalent single-phase circuit (wye-connection).
        Iy_C = I_line / n

        # Actual capacitor current in the delta-connected capacitors.
        Id_C = Iy_C / math.sqrt(3)
        return Id_C


@dataclass
class Transformer:
    """
    Holds the single-phase resistance and (leakage) inductance of the
    transformer.
    """
    R: float = 8.749e-3
    L: float = 26.833e-3 / (2 * math.pi * 50.0)


@dataclass
class Equiv1PhNetwork:
    """
    Represents the equivalent single-phase network of the three-phase network.
    """
    transformer: Transformer
    capacitor_bank: CapacitorBank

    circuit: PassiveElement = field(init=False, default=None)
    # represents the (R + L) // C circuit

    def _make_circuit(self, active_combination: str) -> None:
        self.R = Resistor(self.transformer.R)
        self.L = Inductor(self.transformer.L)
        C_val = self.capacitor_bank.get_C(active_combination)
        self.C = Capacitor(C_val)
        self.circuit = (self.R + self.L) | self.C

    def _solve_at_freq(self, f_hz: float) -> Solution:
        # Get impedance of circuit (R + L) // C.
        Z = Impedance(self.circuit.Z(s=j2pif(f_hz=f_hz)))

        # Build (R + L) // C network.
        nw = Network()
        nw.add_node("0")  # reference node
        nw.add_current_source("I_h", "0", "A", lambda s: 1.0 + 0.0j)  # harmonic current source
        nw.add_short("wire", "A", "B")
        nw.add_passive_element("C", "B", "0", self.C)
        nw.add_passive_element("R", "B", "C", self.R)
        nw.add_passive_element("L", "C", "0", self.L)

        sol = nw.solve_ac(f_hz=f_hz)
        I_C = Phasor(sol.I("C"))
        I_L = Phasor(sol.I("L"))
        U_C = Phasor(sol.V("B"))
        U_L = Phasor(sol.V("C"))

        return Solution(Z, I_C, I_L, U_C, U_L)

    def solve(self, active_combination: str, f_hz: float) -> Solution:
        """
        Solves the network for a harmonic source current of 1 A with the
        specified frequency.

        Parameters
        ----------
        active_combination: str
            Specifies which stages of the capacitor bank are turned on.
        f_hz: float
            Specifies the frequency of the harmonic source current.

        Returns
        -------
        Solution
            Z: Impedance
                The impedance of the (R + L) // C circuit.
            I_C: Phasor
                Line current that flows into the capacitor bank.
            I_L: Phasor
                Line current that flows through the transformer.
            U_C: Phasor
                Phase (line-to-neutral) voltage across the capacitor bank.
            U_L: Phasor
                Phase (line-to-neutral) voltage across the transformer.
        """
        self._make_circuit(active_combination)
        sol = self._solve_at_freq(f_hz)
        return sol


@dataclass(frozen=True)
class Solution:
    Z: Impedance
    I_C: Phasor
    I_L: Phasor
    U_C: Phasor
    U_L: Phasor


def main():
    import numpy as np
    from ent.charts import LineChart

    transformer = Transformer()
    capacitor_bank = CapacitorBank()
    network = Equiv1PhNetwork(transformer, capacitor_bank)

    # For each combination of active stages of the capacitor bank, solve the
    # network for a range of frequencies between 50 (h = 1) and 1250 Hz (h = 25).
    f_hz_range = np.arange(50, 1251, 1)
    h_range = np.asarray([f_hz / 50 for f_hz in f_hz_range])

    soldict = {}
    for comb in capacitor_bank.combinations:
        sols = [network.solve(comb, float(f_hz)) for f_hz in f_hz_range]
        soldict[comb] = sols

    # Plot Z(f_hz) for each combination of active stages.
    ch1 = LineChart()
    for key, sols in soldict.items():
        ch1.add_xy_data(
            label=f"{key}",
            x1_values=h_range,
            y1_values=[sol.Z.mag for sol in sols]
        )
    ch1.x1.scale(h_range[0], h_range[-1] + 1, 1)
    ch1.x1.add_title("harmonic order")
    ch1.y1.add_title("impedance, ohm")
    ch1.add_legend(columns=2)
    ch1.show()

    # Plot capacitor current I_C(f_hz) for each combination of active stages.
    ch2 = LineChart()
    for key, sols in soldict.items():
        ch2.add_xy_data(
            label=f"{key}",
            x1_values=h_range,
            y1_values=[sol.I_C.mag for sol in sols]
        )
    ch2.x1.scale(h_range[0], h_range[-1] + 1, 1)
    ch2.x1.add_title("harmonic order")
    ch2.y1.add_title("line current, A")
    ch2.add_legend(columns=2)
    ch2.show()

    # The capacitor current I_C(f_hz) actually applies to the equivalent
    # single-phase network. It is the line current that flows into the capacitor
    # bank. We now want the true current through a single capacitor.
    I_C_dict = {
        key: [
            capacitor_bank.get_C_current(sol, key)
            for sol in sols
        ] for key, sols in soldict.items()
    }

    ch3 = LineChart()
    for key, sols in I_C_dict.items():
        ch3.add_xy_data(
            label=f"{key}",
            x1_values=h_range,
            y1_values=[I_C for I_C in sols]
        )
    ch3.x1.scale(h_range[0], h_range[-1] + 1, 1)
    ch3.x1.add_title("harmonic order")
    ch3.y1.add_title("capacitor current, A")
    ch3.add_legend(columns=2)
    ch3.show()


if __name__ == '__main__':
    main()
