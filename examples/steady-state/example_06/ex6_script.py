from ent.steady_state import Resistor, Network

nw = Network()
nw.add_node("0")  # reference node

# See ex6_scheme.pdf
nw.add_voltage_source("V1", "0", "A", E=lambda _: -20 + 0j)  # sign E -> V(0) < V(A)
nw.add_passive_element("R1", "A", "B", Resistor(10.0))
nw.add_passive_element("R2", "B", "C", Resistor(5.0))
nw.add_voltage_source("V2", "C", "D", E=lambda _: 15 + 0j)  # sign E -> V(C) > V(D)
nw.add_passive_element("R3", "D", "E", Resistor(10.0))
nw.add_voltage_source("V3", "E", "0", E=lambda _: 5 + 0j)   # sign E -> V(E) > V(0)
nw.add_voltage_source("V4", "0", "F", E=lambda _: 10 + 0j)  # sign E -> V(0) > V(F)
nw.add_passive_element("R4", "F", "B", Resistor(15.0))
nw.add_passive_element("R5", "0", "D", Resistor(20.0))

print(nw)

sol = nw.solve(s=0.0)  # DC -> f = 0

print()

# branch currents
print(f"I(R1) = {sol.I("R1")}")
print(f"I(R4) = {sol.I("R4")}")
print(f"I(R2) = {sol.I("R2")}")
print(f"I(R5) = {sol.I("R5")}")
print(f"I(R3) = {sol.I("R3")}")

print()

# node voltages
print(f"V(B) = {sol.V("B")}")
print(f"V(D) = {sol.V("D")}")


