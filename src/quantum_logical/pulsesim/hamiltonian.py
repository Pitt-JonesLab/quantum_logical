"""Hamiltonian for a quantum system."""

# FIXME after making some changes to the code -
# I am unsure if the docstring equations are all accurate


from quantum_logical.pulsesim.system import QuantumSystem


class Hamiltonian:
    """Hamiltonian for a quantum system."""

    def __init__(self, quantum_system: QuantumSystem, use_RWA=True, use_TLS=True):
        """Initialize the Hamiltonian for a given quantum system.

        Args:
            quantum_system (QuantumSystem): The quantum system for which the Hamiltonian is constructed.
            use_RWA (bool): Flag to indicate whether to use the Rotating Wave Approximation.
            use_TLS (bool): Flag to indicate whether to use the Two-Level System approximation.
        """
        self.system = quantum_system
        self.use_TLS = use_TLS
        self.use_RWA = use_RWA
        self.H = 0

        for mode in self.system.modes:
            self.H += mode.H_0(self.system, RWA=use_RWA, TLS=use_TLS)

        self.H += self._Hint()

    def _Hint(self):
        r"""Generate the coupling part of the Hamiltonian.

        This includes terms for each coupling in the system:
        :math:`H_{\text{int}} = 2 \pi \sum_{\text{couplings}} g_2 (\hat{a}_m + \hat{a}_m^{\dagger})(\hat{a}_n + \hat{a}_n^{\dagger})`
        where \( g_2 \) is the coupling strength between modes \( m \) and \( n \).
        """
        Hint = 0
        for (mode1, mode2), g2 in self.system.couplings.items():
            _field1 = self.system.modes_field[mode1]
            _field2 = self.system.modes_field[mode2]
            Hint += g2 * _field1 * _field2

        return Hint
