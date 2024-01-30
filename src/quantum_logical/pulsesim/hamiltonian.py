"""Hamiltonian for a quantum system."""

# FIXME after making some changes to the code -
# I am unsure if the docstring equations are all accurate

import numpy as np

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
        self.H = (
            self._h0RWA() if (use_RWA or self.use_TLS) else self._h0()
        ) + self._Hint()

    def _h0(self):
        r"""Generate the linear part of the Hamiltonian.

        The linear frequency term for each mode is represented as:
        :math:`H_0 = 2 \pi f \hat{n}`

        For qubit modes, the Hamiltonian includes the linear frequency term and the anharmonicity term:
        :math:`H_0 = 2 \pi [(f - \alpha) \hat{n} + \frac{\alpha}{12} (\hat{a} + \hat{a}^{\dagger})^4]`

        For SNAIL modes, it includes the linear frequency term and the nonlinearity term:
        :math:`H_0 = 2 \pi [f \hat{n} + \frac{g_3}{6} (\hat{a} + \hat{a}^{\dagger})^3]`
        """
        h0 = 0
        for mode in self.system.modes:
            _num = self.system.modes_num[mode]
            _field = self.system.modes_field[mode]

            # Qubit mode
            if hasattr(mode, "alpha"):
                h0 += 2 * np.pi * (mode.freq - mode.alpha) * _num
                h0 += 2 * np.pi * mode.alpha / 12 * _field**4

            # SNAIL mode
            elif hasattr(mode, "g3"):
                h0 += 2 * np.pi * (mode.freq * _num + mode.g3 / 6 * _field**3)

        return h0

    def _h0RWA(self):
        r"""Generate the linear part of the Hamiltonian after applying the RWA.

        For qubit modes:
        :math:`H_{0, \text{RWA}} = 2 \pi [f \hat{n} + \frac{\alpha}{2} \hat{a}^{\dagger} \hat{a}^{\dagger} \hat{a} \hat{a}]`

        For SNAIL modes (preserving only the sss† terms):
        :math:`H_{0, \text{RWA}} = 2 \pi [f \hat{n} + \frac{g_3}{6} (3 \hat{a}^{\dagger} \hat{a} \hat{a} + 3 \hat{a}^{\dagger} \hat{a}^{\dagger} \hat{a})]`
        """
        h0RWA = 0
        for mode in self.system.modes:
            _a = self.system.modes_a[mode]
            _ad = self.system.modes_a_dag[mode]
            _num = self.system.modes_num[mode]
            _field = self.system.modes_field[mode]

            alpha_term = 0
            g3_term = 0

            # FIXME
            if self.use_TLS and hasattr(mode, "alpha"):
                _sz = _ad * _a - _a * _ad
                h0RWA += 2 * np.pi * mode.freq * _sz / 2
                continue

            # Qubit mode
            if hasattr(mode, "alpha"):
                alpha_term = mode.alpha / 2 * _ad * _ad * _a * _a

            # SNAIL mode
            elif hasattr(mode, "g3"):
                # g3_term = mode.g3 / 2 * (_ad * _a * _a + _ad * _ad * _a)
                g3_term = mode.g3 / 6 * (_field**3)

            h0RWA += 2 * np.pi * (mode.freq * _num + alpha_term + g3_term)

        return h0RWA

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
            Hint += 2 * np.pi * g2 * _field1 * _field2

        return Hint
