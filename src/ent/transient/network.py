from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


__all__ = ["Network"]


VoltageWaveform = Callable[[float], float]


@dataclass(frozen=True)
class Node:
    name: str

    def __str__(self) -> str:
        return self.name


class Branch:
    name: str
    n1: Node
    n2: Node


@dataclass(frozen=True)
class ResistorBranch(Branch):
    name: str
    n1: Node
    n2: Node
    R: float  # ohm


@dataclass(frozen=True)
class CapacitorBranch(Branch):
    name: str
    n1: Node
    n2: Node
    C: float         # farad
    v0: float = 0.0  # initial capacitor voltage at t = 0


@dataclass(frozen=True)
class VoltageSourceBranch(Branch):
    name: str
    n1: Node
    n2: Node
    v: VoltageWaveform


@dataclass(frozen=True)
class InductorBranch(Branch):
    name: str
    n1: Node
    n2: Node
    L: float         # henry
    i0: float = 0.0  # initial inductor current (positive n1 -> n2) at t = 0


@dataclass(frozen=True)
class SwitchBranch(Branch):
    name: str
    n1: Node
    n2: Node
    closed: bool = False
    t_start: float | None = None
    R_on: float = 1e-3
    R_off: float = 1e12


@dataclass(frozen=True)
class MNARepresentation:
    """
    Container that holds the data of the MNA-representation of the network. It
    will be used by the network solver to solve for the node voltages and the
    branch currents in the electrical network.
    """
    ref: str
    node_idxmap: dict[str, int]
    voltsources_idxmap: dict[str, int]
    inductor_idxmap: dict[str, int]
    switch_idxmap: dict[str, tuple[int | None, int | None]]
    C: np.ndarray
    G: np.ndarray
    voltsources: tuple[VoltageSourceBranch, ...]
    resistors: tuple[ResistorBranch, ...]
    capacitors: tuple[CapacitorBranch, ...]
    inductors: tuple[InductorBranch, ...]
    switches: tuple[SwitchBranch, ...]


class Network:
    """
    Represents an electrical network for studying transients. This network will
    be solved based on modified nodal analysis (MNA).
    """
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._branches: list[Branch] = []

    def add_node(self, name: str) -> Node:
        if name in self._nodes:
            return self._nodes[name]
        nd = Node(name)
        self._nodes[name] = nd
        return nd

    def default_reference(self) -> str:
        if "0" in self._nodes:
            return "0"
        if "GND" in self._nodes:
            return "GND"
        raise ValueError(
            "No default reference node found ('0' or 'GND'). "
            "Please specify ref=..."
        )

    def add_resistor(
        self,
        name: str,
        n1: str,
        n2: str,
        R: float
    ) -> ResistorBranch:
        """
        Adds a resistor branch to the network.

        Parameters
        ----------
        name: str
            Name to identify the branch in the network.
        n1: str
            Name of the start node of the branch.
        n2: str
            Name of the end node of the branch.
        R: float
            Resistance value in ohm.

        Returns
        -------
        ResistorBranch
        """
        a = self.add_node(n1)
        b = self.add_node(n2)
        br = ResistorBranch(name=name, n1=a, n2=b, R=R)
        self._branches.append(br)
        return br

    def add_capacitor(
        self,
        name: str,
        n1: str,
        n2: str,
        C: float,
        v0: float = 0.0
    ) -> CapacitorBranch:
        """
        Adds a capacitor branch to the network.

        Parameters
        ----------
        name: str
            Name to identify the branch in the network.
        n1: str
            Name of the start node of the branch.
        n2: str
            Name of the end node of the branch.
        C: float
            Capacitor value in Farad.
        v0: float, default 0.0
            Voltage across the capacitor at t = 0.

        Returns
        -------
        CapacitorBranch
        """
        a = self.add_node(n1)
        b = self.add_node(n2)
        br = CapacitorBranch(name=name, n1=a, n2=b, C=C, v0=v0)
        self._branches.append(br)
        return br

    def add_inductor(
        self,
        name: str,
        n1: str,
        n2: str,
        L: float,
        i0: float = 0.0
    ) -> InductorBranch:
        """
        Adds an inductor branch to the network.

        Parameters
        ----------
        name: str
            Name to identify the branch in the network.
        n1: str
            Name of the start node of the branch.
        n2: str
            Name of the end node of the branch.
        L:
            Inductance value in Henry.
        i0:
            Current through the inductor branch t = 0.

        Returns
        -------
        InductorBranch
        """
        a = self.add_node(n1)
        b = self.add_node(n2)
        br = InductorBranch(name=name, n1=a, n2=b, L=L, i0=i0)
        self._branches.append(br)
        return br

    def add_voltage_source(
        self,
        name: str,
        n1: str,
        n2: str,
        v: VoltageWaveform
    ) -> VoltageSourceBranch:
        """
        Adds a voltage source to the network.

        Parameters
        ----------
        name: str
            Name to identify the branch in the network.
        n1: str
            Name of the start node of the branch.
        n2: str
            Name of the end node of the branch.
        v: VoltageWaveform
            A function that takes a time moment t in seconds from t = 0, and
            returns the source voltage at that time moment.

        Returns
        -------
        VoltageSourceBranch
        """
        a = self.add_node(n1)
        b = self.add_node(n2)
        br = VoltageSourceBranch(name=name, n1=a, n2=b, v=v)
        self._branches.append(br)
        return br

    def add_switch(
        self,
        name: str,
        n1: str,
        n2: str,
        closed: bool = False,
        t_start: float | None = None,
        R_on: float = 1e-3,
        R_off: float = 1e12
    ) -> SwitchBranch:
        """
        Adds an idealized switch to the network. Internally, the switch is
        modeled as a resistor with a very low resistance (closed) or very high
        resistance (open).

        Parameters
        ----------
        name: str
            Name to identify the switch in the network.
        n1: str
            Start node of the switch branch.
        n2: str
            End node of the switch branch.
        closed: bool, default False
            Indicates whether the switch is initially closed or not, i.e. at
            t = 0.
        t_start: float, optional
            Time moment (> 0) that the opened switch is closed. Note that closed
            has precedence over t_start, which means that if closed would be
            True, t_start will be ignored.
        R_on: float, default 1e-3
            Equivalent resistance when the switch is closed.
        R_off: float, default 1e12
            Equivalent resistance when the switch is open.

        Returns
        -------
        SwitchBranch
        """
        if t_start is not None and t_start < 0:
            raise ValueError("t_start must be >= 0.")
        if R_on <= 0:
            raise ValueError(f"Switch '{name}': R_on must be > 0.")
        if R_off <= 0:
            raise ValueError(f"Switch '{name}': R_off must be > 0.")

        a = self.add_node(n1)
        b = self.add_node(n2)
        br = SwitchBranch(
            name=name,
            n1=a,
            n2=b,
            closed=closed,
            t_start=t_start,
            R_on=R_on,
            R_off=R_off
        )
        self._branches.append(br)
        return br

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(list(self._nodes.values()))

    @property
    def branches(self) -> tuple[Branch, ...]:
        return tuple(self._branches)

    def compile(
        self,
        ref: str | None = None,
    ) -> MNARepresentation:
        """
        Builds the MNA-representation of the network. The MNA-representation
        contains the building blocks for the differential algebraic equation
        (DAE) in matrix form C * x' + G * x = b(t). This equation is derived by
        applying Kirchoff's current law to each node of the network and by using
        the voltage-current relationship of branches (aka branch constitutive
        equations, BCE).

        Parameters
        ----------
        ref: str, optional
            Indicates the reference node of the network. Note that the reference
            node must already have been added to the network before `compile()`
            is called. If None, a node with the name "0" or "GND" is searched
            for.

        Returns
        -------
        MNARepresentation
            Datacontainer that holds the data of the network MNA-representation
            to be passed to the network solver.
        """
        # Set and add the reference node of the network.
        if ref is None:
            ref = self.default_reference()
        if ref not in self._nodes:
            raise KeyError(f"Reference node '{ref}' not found in network.")

        # Assign an index to each node in the network and map the node names
        # to their index.
        node_names = sorted(self._nodes.keys())
        node_idxmap: dict[str, int] = {}
        k = 0
        for nm in node_names:
            if nm == ref:
                continue
            node_idxmap[nm] = k
            k += 1
        n_nodes = len(node_names) - 1  # number of nodes in the network, except reference node

        # Group the branches of the network by type.
        voltsources = tuple(
            br for br in self._branches
            if isinstance(br, VoltageSourceBranch)
        )
        capacitors = tuple(
            br for br in self._branches
            if isinstance(br, CapacitorBranch)
        )
        resistors = tuple(
            br for br in self._branches
            if isinstance(br, ResistorBranch)
        )
        inductors = tuple(
            br for br in self._branches
            if isinstance(br, InductorBranch)
        )
        switches = tuple(
            br for br in self._branches
            if isinstance(br, SwitchBranch)
        )

        # Assign an index to each voltage source in the network and map the
        # voltage source names to their index.
        voltsource_idxmap = {vs.name: i for i, vs in enumerate(voltsources)}
        n_voltsources = len(voltsources)  # number of voltage sources in the network

        # Assign an index to each inductor in the network and map the inductor
        # names to their index.
        inductor_idxmap = {Lbr.name: i for i, Lbr in enumerate(inductors)}
        n_inductors = len(inductor_idxmap)  # number of inductors in the network

        # Assign a (row, column)-index to each switch in the network and map
        # the switch names to their index.
        switch_idxmap: dict[str, tuple[int | None, int | None]] = {}
        for sw in switches:
            i = None if sw.n1.name == ref else node_idxmap[sw.n1.name]
            j = None if sw.n2.name == ref else node_idxmap[sw.n2.name]
            switch_idxmap[sw.name] = (i, j)

        # Initialize the C and G matrices.
        n = n_nodes + n_voltsources + n_inductors
        C = np.zeros((n, n), dtype=float)
        G = np.zeros((n, n), dtype=float)

        # ----------------------------------------------------------------------
        # Fill-in (stamp) the matrix elements into the matrices C and G
        # ----------------------------------------------------------------------

        def stamp_symmetric(mat: np.ndarray, nm1: str, nm2: str, value: float) -> None:
            n1_is_ref = (nm1 == ref)
            n2_is_ref = (nm2 == ref)

            if not n1_is_ref:
                i = node_idxmap[nm1]
                mat[i, i] += value
            if not n2_is_ref:
                j = node_idxmap[nm2]
                mat[j, j] += value
            if (not n1_is_ref) and (not n2_is_ref):
                i = node_idxmap[nm1]
                j = node_idxmap[nm2]
                mat[i, j] -= value
                mat[j, i] -= value

        # Stamp resistors to G
        for r in resistors:
            if r.R <= 0:
                raise ValueError(f"Resistor '{r.name}' must have R>0.")
            g = 1.0 / r.R
            stamp_symmetric(G, r.n1.name, r.n2.name, g)

        # Stamp switches to G
        for sw in switches:
            R = sw.R_on if sw.closed else sw.R_off
            if R <= 0:
                raise ValueError(f"Switch '{sw.name}' results in non-positive R.")
            g = 1.0 / R
            stamp_symmetric(G, sw.n1.name, sw.n2.name, g)

        # Stamp capacitors into C
        for c in capacitors:
            if c.C < 0:
                raise ValueError(f"Capacitor '{c.name}' must have C>=0.")
            if c.C == 0:
                continue
            # noinspection PyTypeChecker
            stamp_symmetric(C, c.n1.name, c.n2.name, c.C)

        # Stamp voltage sources into G
        for q, vs in enumerate(voltsources):
            row = n_nodes + q

            def stamp_voltsource(nm: str, val: float) -> None:
                if nm == ref:
                    return
                i = node_idxmap[nm]
                G[i, row] += val
                G[row, i] += val

            stamp_voltsource(vs.n1.name, +1.0)
            stamp_voltsource(vs.n2.name, -1.0)

        # Stamp inductors into G and C
        for q, ind in enumerate(inductors):
            if ind.L <= 0:
                raise ValueError(f"Inductor '{ind.name}' must have L > 0.")
            idx_i = n_nodes + n_voltsources + q  # position of i_L in x

            # Stamp inductors into G
            def stamp_inductor(nm: str, val: float) -> None:
                if nm == ref:
                    return
                i = node_idxmap[nm]
                G[i, idx_i] += val
                G[idx_i, i] += val

            stamp_inductor(ind.n1.name, +1.0)
            stamp_inductor(ind.n2.name, -1.0)

            # Stamp inductors into C
            C[idx_i, idx_i] += -ind.L

        return MNARepresentation(
            ref=ref,
            node_idxmap=node_idxmap,
            voltsources_idxmap=voltsource_idxmap,
            inductor_idxmap=inductor_idxmap,
            switch_idxmap=switch_idxmap,
            C=C,
            G=G,
            voltsources=voltsources,
            resistors=resistors,
            capacitors=capacitors,
            inductors=inductors,
            switches=switches
        )
