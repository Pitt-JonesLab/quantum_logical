"""Class for a corral-like coupling scheme."""

from quantum_logical.coupling_util.modular import AbstractModularCoupling


class CorralCoupling(AbstractModularCoupling):
    """A corral-like coupling scheme."""

    def __init__(self, num_snails=8, corral_skip_pattern=(0, 0)):
        """Initialize the corral-like coupling scheme."""
        self.num_snails = num_snails
        self.corral_skip_pattern = corral_skip_pattern
        self.qubit_to_snail_map = {}  # Maps qubits to their primary SNAIL (module)
        super().__init__(description="Corral")

    def _assign_qubit_to_module(self, qubit, snail):
        # Assign a qubit to its first encountered SNAIL neighborhood if it hasn't been assigned yet.
        if qubit not in self.qubit_to_snail_map:
            self.qubit_to_snail_map[qubit] = snail
            # Ensure the module entry exists for the SNAIL
            if snail not in self.modules:
                self.modules[snail] = []
            self.modules[snail].append(qubit)

    def _corral(self, num_snails, skip_pattern):
        num_levels = 2
        assert len(skip_pattern) == num_levels
        snail_edge_list = []
        for snail0, snail1 in zip(range(num_snails), range(1, num_snails + 1)):
            for i in range(num_levels):
                snail_edge_list.append(
                    (snail0, (skip_pattern[i] + snail1) % num_snails)
                )
        return snail_edge_list

    def _snail_to_connectivity(self, snail_edge_list):
        edge_list = []
        for qubit, snail_edge in enumerate(snail_edge_list):
            # Attempt to assign the qubit to a module based on its SNAIL edge
            self._assign_qubit_to_module(qubit, snail_edge[0])
            for temp_qubit, temp_snail_edge in enumerate(snail_edge_list):
                if qubit != temp_qubit and (
                    snail_edge[0] in temp_snail_edge or snail_edge[1] in temp_snail_edge
                ):
                    edge_list.append((qubit, temp_qubit))
        return edge_list

    def _construct_system(self):
        self.snail_edge_list = self._corral(self.num_snails, self.corral_skip_pattern)
        self.c_map = self._snail_to_connectivity(self.snail_edge_list)
        # Each SNAIL is treated as a module for depth purposes, though this could be adjusted
        self.module_depths = {
            snail: snail for snail in range(self.num_snails)
        }  # Example depth assignment
