"""
   Examples of Error Channels:
        - Phase flips
        - Bit flips

    These errors on the hardware qubits propagate into errors in the logical qubits.
    The errors that happen with the highest probability are mapped to error ancilla
    states in the logical encoding. This type of trick, known as an erasure error,
    signals leaving the codeword Hilbert space and is inspired by concepts from
    photonic quantum computing.

"""

from abc import ABC

import numpy as np
from qutip import Qobj, tensor
from qutip.states import basis

from quantum_logical.unitary_util import ImplicitUnitaryGate

zero = basis(2, 0)  # |0>
one = basis(2, 1)  # |1>
p_ge = (basis(2, 0) + basis(2, 1)).unit()  # |+_{ge}>
m_ge = (basis(2, 0) - basis(2, 1)).unit()  # |-_{ge}>
p_gf = (basis(3, 0) + basis(3, 2)).unit()  # |+_{gf}>
m_gf = (basis(3, 0) - basis(3, 2)).unit()  # |-_{gf}>


class LogicalBasis(ABC):
    """Abstract class for a logical basis."""

    def __init__(self, zero_ket, one_ket):
        """Initialize a logical basis.

        Args:
            zero_ket (Qobj): Logical |0> state.
            one_ket (Qobj): Logical |1> state.

        Raises:
            ValueError: If the logical basis states are not orthogonal.
        """
        if not np.isclose(zero_ket.overlap(one_ket), 0):
            raise ValueError("Logical basis states must be orthogonal.")

        self.zero_ket = zero_ket
        self.one_ket = one_ket

    @property
    def transform_operator(self) -> Qobj:
        """Operator that changes from the computational basis to the logical
        basis.

        |0>_L<0| + |1>_L<1|

        Returns:
            Qobj: Transformation operator from computational to logical basis.
        """
        return self.zero_ket * zero.dag() + self.one_ket * one.dag()


class Ancilla:
    """Represents an ancilla qubit for error detection and/or correction.

    Implicitly, ancilla is assumed to be in the |0> until raised to |1> by an error.
    ops= |1_ancilla><error_state|
    or
    op = |1_ancilla><error_state_0| + |1_ancilla><error_state_1| + ...
    for degenerate error states.
    Note, if multiple error states point to same ancilla state, then the ancilla
    state is not able to distinguish between the errors. Then, the error is detectable
    but not correctable.
    """

    # TODO: how should we implement this?
    # TODO: can either be defined by an density state operator, helps us find gate
    # or if we know the gate (for example iswap_ee) we just use that instead?
    # for now, start with abstract operator and rely on transpiler to convert to a gate
    def __init__(self, *error_states: Qobj, correction_operator: Qobj = None):
        """Initialize an ancilla qubit."""
        self.error_states = error_states
        self.correction_operator = correction_operator

    @property
    def detection_operator(self) -> Qobj:
        # |1> \otimes |error_state>  <0| \otimes <error_state|
        return ImplicitUnitaryGate(
            sum(
                [
                    tensor(one, error_state) * tensor(zero, error_state).dag()
                    for error_state in self.error_states
                ]
            )
        )


# could imagine 2 scenarios:
# 1. error detection operator is applied after executing gates
# 2. error detection operator is always being applied (must commute with logical gates)
# second is harder to implement but more advantageous

# TODO: Use Qiskit's error channels?
# https://qiskit.org/ecosystem/aer/apidocs/aer_noise.html#quantum-error-functions


# TODO: how should we implement this?
class LogicalEncoding:
    def __init__(self, logical_basis: LogicalBasis, *ancilla, error_channels=None):
        self.logical_basis = logical_basis
        self.ancilla = ancilla

    def detection_subroutine(self):
        """Generate a quantum circuit subroutine for error detection.

        Returns:
            QuantumCircuit: Subroutine for detecting errors using the ancilla qubits.
        """
        # Implementation details depend on the specific error-detection strategy.
        pass


class DualRail(LogicalEncoding):
    """Dual-rail encoding.

    The logical basis is defined as:
        |0>_L = |01>
        |1>_L = |10>

    Ancilla:
        - |e><00| for detecting photon loss
        - Raise DetectionError (not correctable)

    Error Channels:
        - Amplitude damping
    """

    def __init__(self):
        logical0 = tensor(basis(2, 0), basis(2, 1))
        logical1 = tensor(basis(2, 1), basis(2, 0))
        # to protect against photon loss, define error state as |00>
        loss_state = tensor(basis(2, 0), basis(2, 0))
        double_state = tensor(basis(2, 1), basis(2, 1))
        logical_basis = LogicalBasis(logical0, logical1)
        ancilla = Ancilla(loss_state, double_state)
        super().__init__(logical_basis, ancilla)


class TransmonQuditWithAncilla(LogicalBasis):
    """Transmon qudit encoding with error ancillas.

    The logical basis is defined as:
        |0>_L = |g+f, g+f, g+f>
        |1>_L = |g-f, g-f, g-f>

    Includes ancillas for detecting specific errors.
    """

    def __init__(self):
        logical0 = tensor(p_gf, p_gf, p_gf)
        logical1 = tensor(m_gf, m_gf, m_gf)

        # Define ancillas for error detection and correction
        ancilla1 = Ancilla(
            detection_operator=..., correction_operator=...
        )  # Define appropriately
        ancilla2 = Ancilla(
            detection_operator=..., correction_operator=...
        )  # Define appropriately

        super().__init__(logical0, logical1, ancilla1, ancilla2)


class SNAILConcatWithAncilla(LogicalBasis):
    """SNAIL concatenated erasure encoding with error ancillas.

    The logical basis is defined as:
        |0>_L = |g+f, g+f, g+f, g+f>
        |1>_L = |g-f, g-f, g-f, g-f>

    Includes ancillas for detecting specific errors.
    """

    def __init__(self):
        logical0 = tensor(p_gf, p_gf, p_gf, p_gf)
        logical1 = tensor(m_gf, m_gf, m_gf, m_gf)

        # Define ancillas for error detection and correction
        ancilla1 = Ancilla(
            detection_operator=..., correction_operator=...
        )  # Define appropriately
        ancilla2 = Ancilla(
            detection_operator=..., correction_operator=...
        )  # Define appropriately
        ancilla3 = Ancilla(
            detection_operator=..., correction_operator=...
        )  # Define appropriately

        super().__init__(logical0, logical1, ancilla1, ancilla2, ancilla3)
