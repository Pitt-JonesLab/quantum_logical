"""Class for the Tree scheme of modular coupling."""

from itertools import permutations

from quantum_logical.coupling_util.modular import AbstractModularCoupling


class TreeCoupling(AbstractModularCoupling):
    """A tree-like modular coupling scheme."""

    def __init__(self, module_size=4, children=4, total_levels=3):
        """Initialize the tree-like modular coupling scheme."""
        # Validate children for root module
        assert (
            1 < children <= module_size
        ), "Children must be between 2 and the module size, inclusive for the root module."
        self.module_size = module_size
        self.children = children
        self.total_levels = total_levels
        super().__init__(description="Tree")

    def _construct_system(self):
        # Implement the construction logic specific to the Tree scheme
        # Start the recursive construction of the system
        self._recursive_foo(
            qubit_counter=0, current_level=1, parent_id=None, parent_qubit=None, depth=0
        )

    def _recursive_foo(
        self, qubit_counter, current_level, parent_id, parent_qubit, depth
    ):
        module_id = len(self.modules)  # Unique ID for each module
        self.module_depths[module_id] = depth
        qubits = list(range(qubit_counter, qubit_counter + self.module_size))
        self.modules[module_id] = qubits

        # Create all-to-all connectivity within the module
        edges = list(permutations(qubits, 2))
        self.c_map.extend(edges)

        # Connect this module to its parent module (if applicable)
        if parent_qubit is not None:
            self.c_map.append(
                (parent_qubit, qubits[0])
            )  # Connection from parent to this module
            self.c_map.append((qubits[0], parent_qubit))  # And vice versa

        qubit_counter += self.module_size

        if current_level < self.total_levels:
            child_count = self.children if current_level == 1 else self.children - 1
            for i in range(child_count):
                qubit_counter = self._recursive_foo(
                    qubit_counter,
                    current_level + 1,
                    module_id,
                    qubits[i] if i < child_count - 1 else qubits[-1],
                    depth + 1,
                )

        return qubit_counter
