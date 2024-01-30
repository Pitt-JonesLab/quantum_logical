"""Class representing a single mode in a quantum system."""

import qutip as qt


class QuantumMode:
    """Class representing a single mode in a quantum system."""

    def __init__(self, name: str, freq: float, dim: int, **kwargs):
        """Initialize a QuantumMode instance representing a single mode.

        Args:
            name (str): The name of the quantum mode.
            freq (float): The frequency of the quantum mode.
            dim (int): The dimension of the Hilbert space for this mode.
            **kwargs: Additional properties of the mode, e.g., 'alpha' for qubits, 'g3' for SNAILs.

        The kwargs are used to add any additional attributes specific to different types of quantum modes.
        """
        self.name = name
        self.freq = freq
        self.dim = dim
        self.__dict__.update(kwargs)

        # Initialize quantum operators
        self.a = qt.destroy(dim)  # Annihilation operator
        self.a_dag = self.a.dag()  # Creation operator
        self.num = qt.num(dim)  # Number operator
        self.field = self.a + self.a_dag  # Field operator

        # verify has attribute g3 or alpha, but not both
        assert hasattr(self, "g3") ^ hasattr(self, "alpha")

    def __repr__(self) -> str:
        """Return a string representation of the QuantumMode."""
        return f"QuantumMode(name={self.name}, freq={self.freq} GHz, dim={self.dim})"


# class QubitMode(QuantumMode):
#     def __init__(self, ...):
#         ...

#     def _H0(quantum_system, RWA=True, TLS=True):
#         a, a_dag, num, field = quantum_system.modes_a[self], quantum_system.modes_a_dag[self], quantum_system.modes_num[self], quantum_system.modes_field[self]
#         if not (RWA or TLS):
#             return 2 * np.pi * (mode.freq - mode.alpha) * _num

#         if RWA and not TLS:
#             alpha_term = mode.alpha / 2 * _ad * _ad * _a * _a
#             return 2 * np.pi * (mode.freq * _num + alpha_term)
#         else:
#             _sz = _ad * _a - _a * _ad
#             return 2 * np.pi * mode.freq * _sz / 2

# class CavityMode(QuantumMode):
#     ...

# class SNAILMode(QuantumMode):
#     ...
