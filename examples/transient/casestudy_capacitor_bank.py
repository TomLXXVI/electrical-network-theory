"""
Simulation of transient inrush current when certain combinations of
capacitor bank stages are switched on.
"""
import math
from dataclasses import dataclass, field

from ent.misc import pow_to_imp, delta_to_wye
from ent.transient import Network, sine_waveform_rms, simulate_network
from ent.charts import LineChart


@dataclass
class CapacitorBank:
    Q_stage1: float = 11.845e3 / 3   # reactive power per phase
    Q_stage2: float = 11.845e3 / 3
    Q_stage3: float = 23.689e3 / 3
    Q_stage4: float = 23.689e3 / 3
    V: float = 404.0  # rated voltage across one leg of the capacitor bank.
    omega: float = 2 * math.pi * 50.0

    C_stage1: float = field(init=False, default=0.0)
    C_stage2: float = field(init=False, default=0.0)
    C_stage3: float = field(init=False, default=0.0)
    C_stage4: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.C_stage1 = self._get_C(self.Q_stage1)
        self.C_stage2 = self._get_C(self.Q_stage2)
        self.C_stage3 = self._get_C(self.Q_stage3)
        self.C_stage4 = self._get_C(self.Q_stage4)

    def _get_C(self, Qc: float) -> float:
        """
        Determine the capacitor value C from reactive power Qc.
        """
        # The capacitor impedance value can be derived from reactive power.
        Zcd_val = pow_to_imp(
            phi_deg=-90.0,
            U=self.V,
            Q=Qc
        )
        # The legs of the capacitor bank are connected in delta. Zcd_val is the
        # impedance value of a single capacitor leg in the delta connection.
        # We'll work with a single-phase equivalent network, so we want to
        # replace the delta connection by an equivalent wye connection.
        Zcy_val, *_ = delta_to_wye(Zcd_val)
        C = 1 / (self.omega * abs(Zcy_val))
        return C


@dataclass
class Transformer:
    R: float = 8.700e-3
    L: float = 8.541e-5


@dataclass
class NetworkWrapper:
    capacitor_bank: CapacitorBank
    transformer: Transformer

    network: Network = field(init=False, default=None)

    def __post_init__(self):
        self._build_network()

    def _build_network(self):
        self.network = Network()

        # Ground node
        self.network.add_node("0")  # ground node

        # Voltage source
        self.network.add_voltage_source(
            "V1", "0", "A",
            v=sine_waveform_rms(
                vrms=230.0,
                frequency_hz=50.0,
                phase_rad=math.radians(90.0)
            )
        )
        # Transformer
        self.network.add_resistor("R", "A", "B", self.transformer.R)
        self.network.add_inductor("L", "B", "C", self.transformer.L)

        # Capacitor bank - stage 1
        self.network.add_switch("SW1", "C", "D", closed=True)
        self.network.add_capacitor("C1", "D", "0", self.capacitor_bank.C_stage1)

        # Capacitor bank - stage 2
        self.network.add_switch("SW2", "C", "E", closed=True)
        self.network.add_capacitor("C2", "E", "0", self.capacitor_bank.C_stage2)

        # Capacitor bank - stage 3
        self.network.add_switch("SW3", "C", "F", closed=True)
        self.network.add_capacitor("C3", "F", "0", self.capacitor_bank.C_stage3)

        # Capacitor bank - stage 4
        self.network.add_switch("SW4", "C", "G", closed=False, t_start=0.01)
        self.network.add_capacitor("C4", "G", "0", self.capacitor_bank.C_stage4)

    def simulate(
        self,
        t_end: float = 0.05
    ):
        sol = simulate_network(
            self.network,
            t_end=t_end,
            dt=1e-6,
        )
        return sol


def main():
    capacitor_bank = CapacitorBank()
    transformer = Transformer()
    network = NetworkWrapper(capacitor_bank, transformer)
    sol = network.simulate(t_end=0.05)
    t = sol.t
    i = sol.inductor_current("L")

    ch = LineChart()
    ch.add_xy_data(
        label="current",
        x1_values=t,
        y1_values=i,
    )
    ch.x1.add_title("time, s")
    ch.y1.add_title("current, A")
    ch.show()


if __name__ == '__main__':
    main()
