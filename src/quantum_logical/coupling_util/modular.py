"""This module provides an abstract class for modular coupling maps."""

from abc import ABC, abstractmethod

import rustworkx as rx
from qiskit.transpiler import CouplingMap


class AbstractModularCoupling(CouplingMap, ABC):
    """Abstract class for a modular coupling scheme."""

    def __init__(self, description="ModularCoupling"):
        """Initialize the modular coupling scheme."""
        self.c_map = []  # Temporary, to hold all edges before call to parent
        self.modules = {}  # Maps module ID to its qubits
        self.module_depths = {}  # Maps module ID to its depth
        self.long_edges = []

        # Construct the modular system
        self._construct_system()

        # Remove duplicate edges and filter to keep only edges from lower to higher qubit IDs
        self.c_map = list(set(self.c_map))
        self.c_map = [edge for edge in self.c_map if edge[0] < edge[1]]

        # # Configure gates and measurements based on the generated system
        # gate_configuration = {CXGate: [(i, j) for i, j in self.c_map]}
        # measurable_qubits = list(range(max(max(pair) for pair in self.c_map) + 1))

        super().__init__(self.c_map, description=description)
        self.num_qubits = self.size()  # FIXME

    def add_long_edges(self, num_long_edges=1):
        """Add long-edges to the coupling map."""
        G = rx.PyGraph()
        for node in range(self.num_qubits):
            G.add_node(node)
        for edge in self.c_map:
            G.add_edge(edge[0], edge[1], 1)

        long_edge_endpoints = set()  # Track qubits that are endpoints of long-edges

        for _ in range(num_long_edges):
            all_pairs_lengths = rx.all_pairs_dijkstra_path_lengths(
                G, edge_cost_fn=lambda x: 1
            )

            max_length = 0
            candidates = []
            for source, paths in all_pairs_lengths.items():
                for target, length in paths.items():
                    if source < target and length > max_length:
                        # Check if either qubit is already an endpoint of a long-edge
                        if (
                            source not in long_edge_endpoints
                            and target not in long_edge_endpoints
                        ):
                            max_length = length
                            candidates = [(source, target)]

            # If candidates are found, select the first one as the new long-edge
            if candidates:
                long_edge = candidates[0]
                self.long_edges.append(long_edge)
                G.add_edge(long_edge[0], long_edge[1], 1)
                long_edge_endpoints.update([long_edge[0], long_edge[1]])

    def _find_module_by_qubit(self, qubit):
        """Find the module ID that a given qubit belongs to."""
        for module_id, qubits in self.modules.items():
            if qubit in qubits:
                return module_id
        return None

    @abstractmethod
    def _construct_system(self):
        """Define the edge list.

        Subclasses should implement this method to construct the modular
        system based on their specific scheme.
        """
        pass


# Example usage
# fake_modular = TreeCoupling(module_size=3, children=3, total_levels=3)
# print("Modules:", fake_modular.modules)
# print("Coupling Map:", fake_modular.c_map)
