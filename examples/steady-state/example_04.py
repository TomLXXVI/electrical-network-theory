from ent.steady_state import Network, Resistor, Inductor, Capacitor, j2pif

nw = Network()

nw.add_passive_element("R1", "A", "C", Resistor(10.0))
nw.add_passive_element("L1", "C", "B", Inductor(50e-3))
nw.add_passive_element("C1", "C", "B", Capacitor(100e-6))

# Voltage source: V(A)-V(B) = 10∠0 V
nw.add_voltage_source("V1", "A", "B", E=lambda s: 10.0 + 0j)

sol = nw.solve(ref="B", s=j2pif(50.0))
print(sol)
