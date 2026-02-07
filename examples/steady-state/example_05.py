from ent.steady_state import Network, Resistor

nw = Network()
nw.add_node("0")  # default reference

nw.add_passive_element("R1", "A", "0", Resistor(10.0))
nw.add_voltage_source("V1", "A", "0", E=lambda s: 5.0 + 0j)  # E = V(A) - V(0)
nw.add_current_source("I1", "A", "0", I=lambda s: 1.0 + 0j)  # 1 A from A -> 0

sol = nw.solve_ac(f_hz=50.0)  # ref=None -> uses "0"
print(sol)
