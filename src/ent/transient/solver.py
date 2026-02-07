from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .network import MNARepresentation, Network


__all__ = ["simulate_network"]


def get_initial_state(mna: MNARepresentation, *, tol: float = 1e-9) -> np.ndarray:
    """
    Construct a consistent initial state vector x0 for a time-domain MNA
    simulation.

    This function translates user-defined initial conditions of energy storage
    elements and sources into a state vector that is compatible with the MNA
    formulation

        C · x'(t) + G · x(t) = b(t)

    at t = 0.

    The MNA unknown vector is ordered as:

        x = [ v_nodes , i_vsrc , i_ind ]

    where:
    - v_nodes are the node voltages of all non-reference nodes,
    - i_vsrc are the currents through ideal voltage sources,
    - i_ind are the currents through inductors.

    Notes on initial conditions
    ----------------------------
    * Inductors
      The inductor current is an explicit state variable in MNA. Therefore,
      the initial current i0 of each inductor can be written directly into
      the corresponding entry of x0.

    * Capacitors
      Capacitor voltages are NOT explicit state variables in MNA. Instead,
      capacitors contribute to the C-matrix on node-voltage rows. An initial
      capacitor voltage v0 must therefore be translated into a constraint
      on node voltages:

          v(node1) - v(node2) = v0

    * Ideal voltage sources
      Ideal voltage sources impose algebraic constraints on node voltages.
      At t = 0, each voltage source enforces:

          v(node1) - v(node2) = V_source(0)

      These constraints must already be satisfied by the initial state x0
      in order to start the DAE simulation from a consistent point.

    Implementation strategy
    -----------------------
    All capacitor initial voltages and voltage-source values at t = 0 are
    collected as linear constraints on the node voltages. These constraints
    are solved (in a least-squares sense) to obtain a set of node voltages
    that is algebraically consistent at t = 0.

    If the constraints are mutually inconsistent (e.g. conflicting capacitor
    and source voltages), a ValueError is raised.

    The currents through voltage sources have no prescribed initial value
    and are therefore initialized to zero.

    Parameters
    ----------
    mna : MNARepresentation
        Compiled MNA representation of the network.
    tol : float, optional
        Tolerance used to detect inconsistent initial voltage constraints.

    Returns
    -------
    x0 : numpy.ndarray
        Initial state vector compatible with the MNA formulation, suitable
        for use as the initial condition in a time-domain solver.
    """
    n = mna.C.shape[0]
    x0 = np.zeros((n,), dtype=float)

    n_nodes = len(mna.node_idxmap)
    n_voltsources = len(mna.voltsources_idxmap)

    # ------------------------------------------------------------
    # Build constraints A * v = b for node voltages at t = 0
    # ------------------------------------------------------------
    rows = []
    rhs = []

    def add_voltage_constraint(
        n1: str,
        n2: str,
        value: float,
        label: str
    ) -> None:
        # Equation: v(n1) - v(n2) = value
        row = np.zeros((n_nodes,), dtype=float)

        if n1 != mna.ref:
            row[mna.node_idxmap[n1]] += 1.0
        if n2 != mna.ref:
            row[mna.node_idxmap[n2]] -= 1.0

        # If both ends are reference => 0 = value must hold
        if np.allclose(row, 0.0):
            if abs(value) > tol:
                raise ValueError(
                    f"Inconsistent initial constraint for {label}: "
                    f"both terminals are reference but value={value}."
                )
            return

        rows.append(row)
        rhs.append(float(value))

    if n_nodes > 0:
        # Capacitor initial voltages
        for cap in mna.capacitors:
            add_voltage_constraint(
                cap.n1.name, cap.n2.name, cap.v0,
                label=f"capacitor '{cap.name}' (v0)"
            )

        # Voltage source values at t = 0
        for vs in mna.voltsources:
            v00 = float(vs.v(0.0))
            add_voltage_constraint(
                vs.n1.name, vs.n2.name, v00,
                label=f"voltage source '{vs.name}' (V(0))"
            )

        if rows:
            A = np.vstack(rows)
            b = np.asarray(rhs, dtype=float)

            v_sol, *_ = np.linalg.lstsq(A, b, rcond=None)

            err = A @ v_sol - b
            if np.linalg.norm(err, ord=np.inf) > tol:
                raise ValueError(
                    "Found inconsistent initial conditions (capacitor v0 "
                    "and/or voltage source V(0)). Max constraint error = "
                    f"{np.linalg.norm(err, ord=np.inf):.3e} V"
                )

            x0[:n_nodes] = v_sol

    # ------------------------------------------------------------
    # Inductor initial currents
    # ------------------------------------------------------------
    for ind in mna.inductors:
        q = mna.inductor_idxmap[ind.name]
        idx_i = n_nodes + n_voltsources + q
        x0[idx_i] = ind.i0

    # Voltage source currents have no given initial condition -> keep 0.
    return x0


@dataclass(frozen=True)
class MNASolution:
    """
    Incorporates the solutions of the electrical network in the course of time.

    Attributes
    ----------
    t: np.ndarray
        1D array with the time values
    x: np.ndarray
        2D array with the solutions of the network unknowns.
    """
    network: Network
    mna: MNARepresentation
    t: np.ndarray
    x: np.ndarray

    def node_voltage(self, name: str) -> np.ndarray:
        """
        Returns the voltage of the specified node w.r.t. reference node.
        """
        if name == self.mna.ref:
            return np.zeros_like(self.t)
        i = self.mna.node_idxmap[name]
        return self.x[:, i]

    def voltsource_current(self, name: str) -> np.ndarray:
        """
        Returns the current through the specified voltage source (positive n1->n2).
        """
        n_nodes = len(self.mna.node_idxmap)
        i = self.mna.voltsources_idxmap[name]
        return self.x[:, n_nodes + i]

    def inductor_current(self, name: str) -> np.ndarray:
        """
        Returns the current through the specified inductor.
        """
        n_nodes = len(self.mna.node_idxmap)
        n_voltsources = len(self.mna.voltsources_idxmap)
        i = self.mna.inductor_idxmap[name]
        return self.x[:, n_nodes + n_voltsources + i]

    def __str__(self) -> str:
        # noinspection PyListCreation
        lines = []
        lines.append("TransientSolution")
        lines.append(f"  Reference node: {self.mna.ref}")
        lines.append(f"  Steps: {len(self.t)}")
        lines.append(f"  Unknowns: {self.x.shape[1]}")
        lines.append( "  Nodes: " + ", ".join(sorted([self.mna.ref] + list(self.mna.node_idxmap.keys()))))
        if self.mna.voltsources_idxmap:
            lines.append("  Voltage sources: " + ", ".join(sorted(self.mna.voltsources_idxmap.keys())))
        return "\n".join(lines)


@dataclass
class BackwardEulerMNASolver:
    """
    Numerically solves the differential-algebraic equation (DAE) of an
    electrical network's MNA-representation using backward Euler.

    C * x' + G * x = b(t) => (C/dt + G) * x_{k} = b(t_{k}) + (C/dt) * x_{k-1}
    """
    network: Network
    mna: MNARepresentation

    switch_state: dict[str, bool] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # Track initial switch states (if any)
        for sw in getattr(self.mna, "switches", ()):
            self.switch_state[sw.name] = self._switch_is_closed(sw, 0.0)

    @staticmethod
    def _switch_is_closed(sw, tk: float) -> bool:
        # If t_start is set: close when tk >= t_start
        t_start = getattr(sw, "t_start", None)
        if t_start is not None:
            return tk >= float(t_start)
        return bool(getattr(sw, "closed", False))

    @staticmethod
    def _apply_conductance_delta(
        i: int | None,
        j: int | None,
        dg: float,
        G: np.ndarray,
        A: np.ndarray
    ) -> None:
        if abs(dg) == 0.0:
            return

        # Diagonal contributions
        if i is not None:
            G[i, i] += dg
            A[i, i] += dg
        if j is not None:
            G[j, j] += dg
            A[j, j] += dg

        # Off-diagonal contributions (only if both are non-reference)
        if i is not None and j is not None:
            G[i, j] -= dg
            G[j, i] -= dg
            A[i, j] -= dg
            A[j, i] -= dg

    def _update_switches_if_needed(
        self,
        tk: float,
        G: np.ndarray,
        A: np.ndarray
    ) -> None:
        switches = getattr(self.mna, "switches", ())
        if not switches:
            return

        switch_idxmap = getattr(self.mna, "switch_idxmap", {})
        if not switch_idxmap:
            return

        for sw in switches:
            new_closed = self._switch_is_closed(sw, tk)
            old_closed = self.switch_state.get(sw.name, new_closed)

            if new_closed == old_closed:
                continue  # no change -> no matrix update needed

            # Conductance delta:
            R_on = float(getattr(sw, "R_on", 1e-3))
            R_off = float(getattr(sw, "R_off", 1e12))
            g_new = 1.0 / (R_on if new_closed else R_off)
            g_old = 1.0 / (R_on if old_closed else R_off)
            dg = g_new - g_old

            i, j = switch_idxmap[sw.name]
            self._apply_conductance_delta(i, j, dg, G, A)

            self.switch_state[sw.name] = new_closed

    def simulate(
        self,
        t_end: float,
        dt: float,
        x0: np.ndarray | None = None
    ) -> MNASolution:
        """
        Numerically solves the DAE at each time moment between t = 0 and t_end
        using backward Euler.

        Parameters
        ----------
        t_end: float
            Final time moment (s) at which the simulation stops.
        dt: float
            Time step (s) between two successive moments.
        x0: np.ndarray, optional
            Initial conditions at t = 0.
        """
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        if t_end <= 0:
            raise ValueError("t_end must be > 0.")

        C = self.mna.C
        G = self.mna.G
        n = C.shape[0]

        n_steps = int(np.floor(t_end / dt)) + 1
        t = np.linspace(0.0, dt * (n_steps - 1), n_steps)

        x = np.zeros((n_steps, n), dtype=float)

        if x0 is not None:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (n,):
                raise ValueError(f"x0 must have shape ({n},)")
            x[0, :] = x0

        A = (C / dt) + G

        # Helper: build b(t_{k}) (only voltage sources for now)
        def b_of_t(tk: float) -> np.ndarray:
            rhs = np.zeros((n,), dtype=float)
            n_nodes = len(self.mna.node_idxmap)
            for i, vs in enumerate(self.mna.voltsources):
                rhs[n_nodes + i] = float(vs.v(tk))
            return rhs

        for k in range(1, n_steps):
            tk = t[k]
            self._update_switches_if_needed(tk, G, A)
            rhs = b_of_t(tk) + (C / dt) @ x[k - 1, :]
            try:
                x[k, :] = np.linalg.solve(A, rhs)
            except np.linalg.LinAlgError as e:
                raise RuntimeError("Transient MNA matrix is singular at this step.") from e

        return MNASolution(network=self.network, mna=self.mna, t=t, x=x)


def simulate_network(
    nw: Network,
    t_end: float,
    dt: float
) -> MNASolution:
    """
    Solves an electrical network using Modified Nodal Analysis (MNA) and
    backward Euler. The network is solved at each time moment between t = 0 and
    t_end. Results are accessible through the returned `MNASolution` object.

    Parameters
    ----------
    nw: Network
        Electrical network to be solved.
    t_end: float
        Final time moment (s) at which the simulation stops.
    dt: float
        Time step (s) between two successive moments.

    Returns
    -------
    MNASolution
    """
    mna_repr = nw.compile()  # build MNA representation of network
    solver = BackwardEulerMNASolver(network=nw, mna=mna_repr)  # initialize solver
    x0 = get_initial_state(mna_repr)  # get initial conditions in the network
    sol = solver.simulate(t_end=t_end, dt=dt, x0=x0)  # run simulation
    return sol
