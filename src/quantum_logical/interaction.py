"""Conversion and gain interaction Hamiltonian."""

import numpy as np
import qutip
from qutip import tensor

from quantum_logical.deprecated.operators import selective_destroy

__all__ = [
    "ConversionGainInteraction",
    "ConversionGainFiveWave",
    "CNOT_FC_EF",
]


class Hamiltonian:
    """Hamiltonian for a quantum system."""

    def __init__(self, coefficients, operators):
        """Initialize the Hamiltonian.

        Initialize the Hamiltonian object with given coefficients and
        operators, automatically including Hermitian conjugates when necessary,
        and avoiding duplication if the Hermitian conjugate is already
        provided.

        Args:
            coefficients (list of complex or qutip.Qobj): Corresponding coefficients.
            operators (list of qutip.Qobj): List of operators.
        """
        self.coefficients = list(coefficients)
        self.operators = list(operators)

        # Add Hermitian conjugates only if they're not already in the list
        for coeff, op in zip(coefficients, operators):
            if not op.isherm:
                op_dag = op.dag()
                if not any(
                    (op_dag - existing_op).norm() < 1e-10
                    for existing_op in self.operators
                ):
                    self.coefficients.append(np.conjugate(coeff))
                    self.operators.append(op_dag)

        self.operator = self.construct_operator()

    def construct_operator(self):
        """Construct the Hamiltonian matrix from coefficients and operators."""
        return sum(coeff * op for coeff, op in zip(self.coefficients, self.operators))

    def construct_U(self, t: float) -> qutip.Qobj:
        """Construct the unitary evolution operator for time t.

        Args:
            t (float): Time for which the unitary evolution is calculated.

        Returns:
            qutip.Qobj: Unitary operator for the given time.
        """
        return (-1j * self.operator * t).expm()


class PairwiseInteractionMixin:
    """Mixin class for pairwise interactions."""

    @staticmethod
    def verify_pairwise_interaction(ops):
        """Verify that each term involves exactly two qubits."""
        for op in ops:
            if len(op.dims[0]) != 2:
                raise ValueError("Each term must involve exactly two qubits.")


class ConversionGainInteraction(Hamiltonian, PairwiseInteractionMixin):
    """Conversion and gain interaction Hamiltonian."""

    def __init__(self, gc, gg, phi_c=0.0, phi_g=0.0, transmon_levels=2):
        """Initialize the interaction with given parameters."""
        # Initialize qutip identity and annihilation operator for n-level system
        a_ge = selective_destroy(transmon_levels, 1, 0)

        # Define the interaction terms and coefficients
        coefficients = [gc * np.exp(1j * phi_c), gg * np.exp(1j * phi_g)]
        operators = [
            tensor(a_ge, a_ge.dag()),  # Conversion term
            tensor(a_ge, a_ge),  # Gain term
        ]

        # Validate pairwise interactions
        self.verify_pairwise_interaction(operators)

        # Initialize the Hamiltonian
        super().__init__(coefficients, operators)

    @classmethod
    def from_coeff_ops_list(cls, coefficients, operators):
        """Initialize the interaction with given coefficients and operators.

        Args:
            coefficients (list): Coefficients for the Hamiltonian terms.
            operators (list): Operators for the Hamiltonian terms.

        Returns:
            ConversionGainInteraction: An instance of the class.
        """
        # Validate pairwise interactions
        PairwiseInteractionMixin.verify_pairwise_interaction(operators)

        # Initialize the Hamiltonian directly
        return Hamiltonian(coefficients, operators)


# TODO add all 3wave interactions as a class


class ConversionGainFiveWave(Hamiltonian, PairwiseInteractionMixin):
    """Conversion and gain interaction Hamiltonian."""

    def __init__(self, gc, gg, phi_c=0.0, phi_g=0.0, transmon_levels=3):
        """Initialize the interaction with given parameters."""
        # Define annihilation and creation operators for specified transitions
        a_gf = selective_destroy(transmon_levels, 2, 0)

        # Define the interaction terms and coefficients
        coefficients = [
            gc * np.exp(1j * phi_c),
            gg * np.exp(1j * phi_g),
        ]
        operators = [
            tensor(a_gf, a_gf.dag()),  # a_gf * b_gf† term
            tensor(a_gf, a_gf),  # a_gf * b_gf term
        ]

        # Validate pairwise interactions
        self.verify_pairwise_interaction(operators)

        # Initialize the Hamiltonian
        super().__init__(coefficients, operators)


class CNOT_FC_EF(Hamiltonian, PairwiseInteractionMixin):
    """CNOT controlled by |f>, target gate is between |e> and |f>."""

    def __init__(self, gc, gg, phi_c=0.0, phi_g=0.0, transmon_levels=3):
        """Initialize the interaction with given parameters."""
        # Define annihilation and creation operators for specified transitions
        a_ef = selective_destroy(transmon_levels, 2, 1)

        # Define the interaction terms and coefficients
        coefficients = [
            gc * np.exp(1j * phi_c),
            gg * np.exp(1j * phi_g),
        ]
        operators = [
            tensor(a_ef, a_ef.dag()),  # a_ef * a_ef term
            tensor(a_ef, a_ef),  # a_ef * a_ef term
        ]

        # Validate pairwise interactions
        self.verify_pairwise_interaction(operators)

        # Initialize the Hamiltonian
        super().__init__(coefficients, operators)


if __name__ == "__main_":
    # Example usage:
    H = ConversionGainInteraction(np.pi / 4, np.pi / 4)
    print(H.construct_U(1.0))
