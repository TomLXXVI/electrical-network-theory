from __future__ import annotations

import math
from typing import Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..elements.base import PassiveElement

__all__ = ["Network"]


@dataclass(frozen=True)
class Node:
    name: str

    def __str__(self) -> str:
        return self.name


class Branch(ABC):
    name: str
    n1: Node
    n2: Node

    @abstractmethod
    def is_voltage_source(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class PassiveBranch(Branch):
    """
    Represents a branch with a passive element (resistor, inductor, or
    capacitor).

    Attributes
    ----------
    name: str
        Name to identify the branch in the network.
    n1: Node
        Start node of the branch.
    n2: Node
        End node of the branch.
    elem: PassiveElement
        Impedance of the passive element.
    """
    name: str
    n1: Node
    n2: Node
    elem: PassiveElement

    def is_voltage_source(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"{self.name}: {self.n1}-{self.n2}  {self.elem}"


VoltageValue = Callable[[complex], complex]  # E(s) in Laplace/phasor domain


@dataclass(frozen=True)
class VoltageSourceBranch(Branch):
    """
    Represents an ideal independent voltage source E(s).

    Attributes
    ----------
    name: str
        Name to identify the branch in the network.
    n1: Node
        Start node of the branch.
    n2: Node
        End node of the branch.
    E: VoltageValue
        Function that takes s, and returns the voltage E(s).
    """
    name: str
    n1: Node
    n2: Node
    E: VoltageValue

    def is_voltage_source(self) -> bool:
        return True

    def __str__(self) -> str:
        return f"{self.name}: {self.n1}-{self.n2}  Vs"


CurrentValue = Callable[[complex], complex]  # I(s) in Laplace/phasor domain


@dataclass(frozen=True)
class CurrentSourceBranch(Branch):
    """
    Represents a current source.

    Attributes
    ----------
    name: str
        Name to identify the branch in the network.
    n1: Node
        Start node of the branch.
    n2: Node
        End node of the branch.
    I: CurrentValue
        A function that takes s, and returns the current I(s).
    """
    name: str
    n1: Node
    n2: Node
    I: CurrentValue

    def is_voltage_source(self) -> bool:
        return False

    def __str__(self) -> str:
        return f"{self.name}: {self.n1}-{self.n2} Is"


class Network:
    """
    Represents an electrical network.
    """
    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._branches: list[Branch] = []

    def default_reference(self) -> str:
        if "0" in self._nodes:
            return "0"
        if "GND" in self._nodes:
            return "GND"
        raise ValueError("No default reference node found ('0' or 'GND'). Please specify ref=...")

    def add_node(self, name: str) -> Node:
        if name in self._nodes:
            return self._nodes[name]
        node = Node(name=name)
        self._nodes[name] = node
        return node

    def add_passive_element(self, name: str, n1: str, n2: str, elem: PassiveElement) -> PassiveBranch:
        """
        Adds a new branch with a passive element to the network.

        Parameters
        ----------
        name: str
            Name to identify the branch in the network.
        n1: str
            Name of the start node of the branch.
        n2: str
            Name of the end node of the branch.
        elem: PassiveElement
            Impedance of the passive element.
        """
        node1 = self.add_node(n1)
        node2 = self.add_node(n2)
        br = PassiveBranch(name=name, n1=node1, n2=node2, elem=elem)
        self._branches.append(br)
        return br

    def add_voltage_source(self, name: str, n1: str, n2: str, E: VoltageValue) -> VoltageSourceBranch:
        """
        Adds a new branch with a voltage source to the network.

        Parameters
        ----------
        name: str
            Name to identify the branch in the network.
        n1: str
            Name of the start node of the branch.
        n2: str
            Name of the end node of the branch.
        E: VoltageValue
            Function that takes s, and returns the voltage E(s).
        """
        node1 = self.add_node(n1)
        node2 = self.add_node(n2)
        br = VoltageSourceBranch(name, node1, node2, E=E)
        self._branches.append(br)
        return br

    def add_current_source(self, name: str, n1: str, n2: str, I: CurrentValue) -> CurrentSourceBranch:
        """
        Adds a new branch with a current source to the network.

        Parameters
        ----------
        name: str
            Name to identify the branch in the network.
        n1: str
            Name of the start node of the branch.
        n2: str
            Name of the end node of the branch.
        I: CurrentValue
            Function that takes s, and returns the current I(s).
        """
        node1 = self.add_node(n1)
        node2 = self.add_node(n2)
        br = CurrentSourceBranch(name, node1, node2, I)
        self._branches.append(br)
        return br

    def add_short(self, name: str, n1: str, n2: str) -> VoltageSourceBranch:
        """
        Adds an ideal short circuit (0 V constraint) between nodes n1 and n2.

        An ideal short is represented as an ideal 0 V voltage source enforcing
        V(n1) = V(n2). The current through the short is returned as the source
        current.
        """
        return self.add_voltage_source(name=name, n1=n1, n2=n2, E=lambda s: 0.0 + 0j)

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(list(self._nodes.values()))

    @property
    def branches(self) -> tuple[Branch, ...]:
        return tuple(self._branches)

    def __str__(self) -> str:
        # noinspection PyListCreation
        lines = ["Network:"]
        lines.append("  Nodes: " + ", ".join(sorted(self._nodes.keys())))
        lines.append("  Branches:")
        for br in self._branches:
            lines.append(f"    - {br}")
        return "\n".join(lines)

    def _build_ybus(self, s: complex, ref: Node) -> tuple[np.ndarray, dict[str, int]]:
        """
        Builds the reduced nodal admittance matrix (reference node removed).

        Returns
        -------
        tuple[np.ndarray, dict[str, int]]
            Y_red:
                (N-1)x(N-1) complex matrix
            node_idxmap:
                mapping node_name -> row/col index in reduced matrix (ref not
                included)
        """
        node_names = sorted(self._nodes.keys())
        if ref.name not in self._nodes:
            raise KeyError(f"Reference node '{ref.name}' is not in the network.")

        # reduced indexing: skip reference node
        node_idxmap: dict[str, int] = {}
        k = 0
        for nm in node_names:
            if nm == ref.name:
                continue
            node_idxmap[nm] = k
            k += 1

        n_red = len(node_names) - 1
        Y = np.zeros((n_red, n_red), dtype=complex)

        def add_stamp(nm_i: str, nm_j: str, y: complex) -> None:
            """Stamp a branch admittance y between nodes i and j into reduced Y."""
            i_is_ref = (nm_i == ref.name)
            j_is_ref = (nm_j == ref.name)

            if not i_is_ref:
                ii = node_idxmap[nm_i]
                Y[ii, ii] += y
            if not j_is_ref:
                jj = node_idxmap[nm_j]
                Y[jj, jj] += y
            if (not i_is_ref) and (not j_is_ref):
                ii = node_idxmap[nm_i]
                jj = node_idxmap[nm_j]
                Y[ii, jj] -= y
                Y[jj, ii] -= y

        for br in self._branches:
            # noinspection PyUnresolvedReferences
            z = br.elem.Z(s)
            if z == 0:
                raise ZeroDivisionError(
                    f"Branch '{br.name}' has Z(s)=0 at s={s}; "
                    "ideal short not supported in Y-bus stamping yet."
                )
            y = 1 / z
            add_stamp(br.n1.name, br.n2.name, y)

        return Y, node_idxmap

    def solve_with_current_injection(
        self,
        a: str,
        b: str,
        Iinj: complex,
        s: complex
    ) -> NetworkSolution:
        """
        Solves the **elements** network between node a and node b.

        A current +Iinj is injected into node a, while node b is connected to
        ground (0 V). Returns the node voltages and branch currents in the
        network between node a and node b.

        Returns
        -------
        NetworkSolution

        Raises
        ------
        ValueError
            If the network also contains voltage source branches.
        """
        if any(isinstance(br, VoltageSourceBranch) for br in self._branches):
            raise ValueError(
                "solve_with_current_injection() is only valid for "
                "elements-only networks. This network contains "
                "voltage sources. Use solve_with_voltage_sources(...)."
            )

        if a == b:
            raise ValueError("Injection nodes must be different.")
        if a not in self._nodes or b not in self._nodes:
            raise KeyError("Both nodes must exist in the network.")

        node_a = self._nodes[a]
        node_b = self._nodes[b]  # reference

        Y, idx = self._build_ybus(s=s, ref=node_b)

        I = np.zeros((Y.shape[0],), dtype=complex)
        I[idx[node_a.name]] += Iinj

        try:
            V_red = np.linalg.solve(Y, I)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "Network admittance matrix is singular. "
                "Are the nodes disconnected from the reference?"
            ) from e

        # Build full node voltage dict including reference node at 0
        voltages: dict[str, complex] = {node_b.name: 0.0 + 0j}
        for nm, k in idx.items():
            voltages[nm] = V_red[k]

        # Compute branch currents with sign convention n1 -> n2
        currents: dict[str, complex] = {}
        for br in self._branches:
            v1 = voltages[br.n1.name]
            v2 = voltages[br.n2.name]
            # noinspection PyUnresolvedReferences
            z = br.elem.Z(s)
            if z == 0:
                raise ZeroDivisionError(
                    f"Branch '{br.name}' has Z(s)=0 at s={s}; "
                    "ideal short not supported in current calculation yet."
                )
            currents[br.name] = (v1 - v2) / z

        return NetworkSolution(
            voltages=voltages,
            currents=currents,
            reference=node_b.name,
            network=self
        )

    def equivalent_impedance(self, a: str, b: str, s: complex) -> complex:
        """
        Returns the equivalent network elements Z_eq between nodes a and b.

        Method: inject +1 A into node a, while node b is connected to ground
        (0 V), solve Y*V = I, then Z_eq = (V_a - V_b)/1A = V_a.
        """
        sol = self.solve_with_current_injection(a, b, Iinj=1.0 + 0j, s=s)
        return sol.V(a)

    def solve(
        self,
        s: complex,
        ref: str | None = None,
        injections: dict[str, complex] | None = None,
    ) -> NetworkSolution:
        """
        Solves the network in the frequency (Laplace) domain using
        Modified Nodal Analysis (MNA).

        This solver supports:
        -   passive elements branches (R/L/C or composed impedances),
        -   ideal independent voltage sources,
        -   ideal independent current sources,
        -   optional explicit node current injections.

        Node voltages are returned with respect to the reference node 'ref'
        (V(ref)=0). Branch currents are returned with the convention that
        positive current flows from n1 -> n2.

        Parameters
        ----------
        s: complex
            Complex frequency variable. For sinusoidal steady-state at
            frequency f (Hz), use s = j * 2*pi*f.
        ref: str, optional
            Indicates the reference node of the network. Note that the reference
            node must already have been added to the network before `solve()`
            is called. If None, a node with the name "0" or "GND" is searched
            for.
        injections: dict, optional
            Current injection map of the form {node_name: Iinj}. A positive
            value injects current into the node (i.e., it appears on the
            right-hand side of KCL). Injections at the reference node are
            ignored in the reduced system.

        Voltage source convention
        -------------------------
        For a voltage source branch defined from n1 -> n2 with source value E(s),
        the enforced constraint is:

            V(n1) - V(n2) = E(s)

        The returned current for that voltage source branch is the corresponding
        MNA unknown, and it is reported with the same sign convention:

            I(source) > 0  means current flowing from n1 to n2.

        Impedance branch current convention
        -----------------------------------
        For an elements branch with elements Z(s) between n1 and n2, the
        reported current is:

            I(branch) = (V(n1) - V(n2)) / Z(s)

        Returns
        -------
        NetworkSolution
            A solution object containing node voltages and branch currents.

        Raises
        ------
        ValueError, KeyError
            If the reference node does not exist.
        ZeroDivisionError
            If any elements branch yields Z(s) = 0 at the provided 's' (ideal
            shorts not supported yet).
        RuntimeError
            If the MNA system matrix is singular (e.g., floating subcircuits or
            inconsistent constraints).
        """
        if ref is None:
            ref = self.default_reference()
        if ref not in self._nodes:
            raise KeyError(f"Reference node '{ref}' not found in network.")

        # --- incorporate current source branches into injections ---
        # Convention: current flows from n1 -> n2 with value I(s)
        # => node n1 gets -I(s), node n2 gets +I(s)
        injections = injections or {}
        for br in self._branches:
            if isinstance(br, CurrentSourceBranch):
                Ival = br.I(s)
                injections[br.n1.name] = injections.get(br.n1.name, 0j) - Ival
                injections[br.n2.name] = injections.get(br.n2.name, 0j) + Ival

        # --- index maps for non-reference nodes ---
        node_names = sorted(self._nodes.keys())
        node_idxmap: dict[str, int] = {}
        k = 0
        for nm in node_names:
            if nm == ref:
                continue
            node_idxmap[nm] = k
            k += 1
        n_nodes = len(node_names) - 1

        # --- collect voltage sources ---
        voltsources: list[VoltageSourceBranch] = [b for b in self._branches if isinstance(b, VoltageSourceBranch)]
        n_voltsources = len(voltsources)
        # idx_s: dict[str, int] = {vs.name: i for i, vs in enumerate(voltsources)}

        # --- build MNA matrix ---
        A = np.zeros((n_nodes + n_voltsources, n_nodes + n_voltsources), dtype=complex)
        rhs = np.zeros((n_nodes + n_voltsources,), dtype=complex)

        # Stamp passive admittances into Y (upper-left block)
        def stamp_admittance(nm1: str, nm2: str, y: complex) -> None:
            n1_is_ref = (nm1 == ref)
            n2_is_ref = (nm2 == ref)

            if not n1_is_ref:
                i = node_idxmap[nm1]
                A[i, i] += y
            if not n2_is_ref:
                j = node_idxmap[nm2]
                A[j, j] += y
            if (not n1_is_ref) and (not n2_is_ref):
                i = node_idxmap[nm1]
                j = node_idxmap[nm2]
                A[i, j] -= y
                A[j, i] -= y

        for br in self._branches:
            if isinstance(br, PassiveBranch):
                z = br.elem.Z(s)
                if z == 0:
                    raise ZeroDivisionError(
                        f"Branch '{br.name}' has Z(s)=0 at s={s}; "
                        f"ideal short not supported yet."
                    )
                y = 1 / z
                stamp_admittance(br.n1.name, br.n2.name, y)

        # Stamp current injections into rhs (top part)
        for nm, Iinj in injections.items():
            if nm == ref:
                # Injecting current into reference is allowed but doesn't appear in reduced system
                continue
            if nm not in node_idxmap:
                raise KeyError(f"Injection node '{nm}' not found.")
            rhs[node_idxmap[nm]] += Iinj

        # Stamp voltage sources into B / B^T and E into rhs bottom
        # Convention: V(n1) - V(n2) = E(s)
        for q, vs in enumerate(voltsources):
            # B stamps into upper-right and lower-left
            def add_B(node_name: str, value: complex) -> None:
                if node_name == ref:
                    return
                i = node_idxmap[node_name]
                A[i, n_nodes + q] += value      # upper-right
                A[n_nodes + q, i] += value      # lower-left (B^T)

            add_B(vs.n1.name, +1.0 + 0j)
            add_B(vs.n2.name, -1.0 + 0j)

            rhs[n_nodes + q] = vs.E(s)  # E(s)

        # Solve
        try:
            x = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "MNA matrix is singular. Check connectivity and sources."
            ) from e

        V_red = x[:n_nodes]
        I_src = x[n_nodes:]  # currents through voltage sources, sign per MNA convention

        # Build voltage dict including ref = 0
        voltages: dict[str, complex] = {ref: 0.0 + 0j}
        for nm, i in node_idxmap.items():
            voltages[nm] = V_red[i]

        # Compute currents for all branches
        currents: dict[str, complex] = {}

        # Impedance branch currents: positive n1 -> n2
        for br in self._branches:
            if isinstance(br, PassiveBranch):
                v1 = voltages[br.n1.name]
                v2 = voltages[br.n2.name]
                currents[br.name] = (v1 - v2) / br.elem.Z(s)
            if isinstance(br, CurrentSourceBranch):
                currents[br.name] = br.I(s)

        # Voltage source currents: these are direct unknowns in MNA
        # Convention: current variable is positive from n1 -> n2 (consistent with V(n1)-V(n2)=E)
        for q, vs in enumerate(voltsources):
            currents[vs.name] = I_src[q]

        return NetworkSolution(network=self, voltages=voltages, currents=currents, reference=ref)

    def solve_ac(
        self,
        f_hz: float,
        ref: str | None = None,
        injections: dict[str, complex] | None = None
    ) -> NetworkSolution:
        """
        Solves the network at sinusoidal steady-state frequency f_hz using MNA.

        Parameters
        ----------
        f_hz:
            Steady-state frequency in Hz.
        ref:
            Reference node of the network (0 V).
        injections:
            Optional current injection map {node_name: Iinj}. A positive value
            injects current into the node (i.e., it appears on the right-hand
            side of KCL). Injections at the reference node are ignored in the
            reduced system.

        Returns
        -------
        NetworkSolution
        """
        s = 1j * 2.0 * math.pi * f_hz
        return self.solve(s, ref, injections)


@dataclass(frozen=True)
class NetworkSolution:
    """
    Solution of a frequency-domain network problem.

    Attributes
    ----------
    voltages: dict[str, complex]
        node_name -> complex voltage (V) w.r.t. reference node (0V).
    currents: dict[str, complex]
        branch_name -> complex current (A), positive from branch.n1 to branch.n2
    reference: str
        reference node name
    """
    voltages: dict[str, complex]
    currents: dict[str, complex]
    reference: str
    network: Network

    def V(self, node: str) -> complex:
        """Returns the node voltage."""
        return self.voltages[node]

    def I(self, branch: str) -> complex:
        """Returns the branch current."""
        return self.currents[branch]

    def __str__(self) -> str:
        lines: list[str] = [
            "Network solution",
            f"  Reference node: {self.reference}",
            "",
            "  Node voltages:"
        ]
        for node in sorted(self.voltages.keys()):
            v = self.voltages[node]
            lines.append(f"    V({node}) = {v:.6g} V")

        lines.append("")
        lines.append("  Branch currents (positive n1 -> n2):")
        for br in self.network.branches:
            i = self.currents[br.name]
            lines.append(f"  I({br.name}) [{br.n1}->{br.n2}] = {i:.6g} A")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            "NetworkSolution("
            f"voltages={self.voltages!r}, "
            f"currents={self.currents!r}, "
            f"reference={self.reference!r})"
        )
