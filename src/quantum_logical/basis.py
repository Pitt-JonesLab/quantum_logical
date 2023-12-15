"""Logical encoding of qubits.

Examples of Error Channels:
    - Phase flips
    - Bit flips

These errors on the hardware qubits propagate into errors in the logical qubits.
The errors that happen with the highest probability are mapped to error ancilla
states in the logical encoding. This type of trick, known as an erasure error,
signals leaving the codeword Hilbert space and is inspired by concepts from
photonic quantum computing.
"""

from abc import ABC, abstractmethod

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, HGate, IGate, XGate, ZGate
from qutip import Qobj
from qutip.states import basis
from qutip.tensor import tensor

from quantum_logical.operators import transform_ge_to_gf_gate
from quantum_logical.qudit_op import QutritUnitary

# zero = basis(2, 0)  # |0>
# one = basis(2, 1)  # |1>
e = basis(3, 1)  # |e>
p_ge = (basis(2, 0) + basis(2, 1)).unit()  # |+_{ge}>
m_ge = (basis(2, 0) - basis(2, 1)).unit()  # |-_{ge}>
p_gf = (basis(3, 0) + basis(3, 2)).unit()  # |+_{gf}>
m_gf = (basis(3, 0) - basis(3, 2)).unit()  # |-_{gf}>

x_gf = transform_ge_to_gf_gate(XGate().to_matrix())
x_gf = QutritUnitary(x_gf, "x_gf")
cx_gf = transform_ge_to_gf_gate(CXGate().to_matrix())
cx_gf = QutritUnitary(cx_gf, "cx_gf")
h_gf = transform_ge_to_gf_gate(HGate().to_matrix())
h_gf = QutritUnitary(h_gf, "h_gf")


class QuantumErrorCorrectionCode(ABC):
    """Abstract class for logical encoding in quantum error correction.

    This class provides the framework for transforming logical qubits
    into encoded qubits, essential for implementing error correction
    codes.
    """

    code_length: int = None  # Number of qubits that define the codewords
    num_ancilla: int = None
    stabilizers: list = None

    def __init__(self, encoded_zero_ket: Qobj, encoded_one_ket: Qobj):
        """Initialize the logical encoding with encoded basis states.

        Args:
            encoded_zero_ket (Qobj): The ket representing the encoded logical 0.
            encoded_one_ket (Qobj): The ket representing the encoded logical 1.

        Raises:
            ValueError: If the encoded states are not orthogonal or not properly normalized.
        """
        if not self._are_states_valid(encoded_zero_ket, encoded_one_ket):
            raise ValueError("Encoded states must be orthogonal and normalized.")

        self.encoded_zero_ket = encoded_zero_ket
        self.encoded_one_ket = encoded_one_ket

    @staticmethod
    def _are_states_valid(state0: Qobj, state1: Qobj) -> bool:
        """Check if the provided states are orthogonal and normalized.

        Args:
            state0 (Qobj): First quantum state.
            state1 (Qobj): Second quantum state.

        Returns:
            bool: True if states are valid, False otherwise.
        """
        return (
            np.isclose(state0.norm(), 1)
            and np.isclose(state1.norm(), 1)
            and np.isclose(state0.overlap(state1), 0)
        )

    @property
    def transform_operator(self) -> Qobj:
        """Transform from the computational (logical) to the encoded basis.

        Returns:
            Qobj: Transformation operator from computational to encoded basis.
        """
        return (
            self.encoded_zero_ket * basis(2, 0).dag()
            + self.encoded_one_ket * basis(2, 1).dag()
        )

    @property
    def projector(self) -> Qobj:
        """Projector onto the encoded basis.

        Returns:
            Qobj: Projector operator.
        """
        return (
            self.encoded_zero_ket * self.encoded_zero_ket.dag()
            + self.encoded_one_ket * self.encoded_one_ket.dag()
        )

    def stabilizer_subroutine(
        self, qc, code_register, ancilla_register, ancilla_classical_register
    ):
        """Insert the stabilizer subroutine into the given quantum circuit.

        Modifies the given quantum circuit by adding the stabilizer
        subroutine, including syndrome extraction and correction.

        Args:
            qc (QuantumCircuit): The quantum circuit to modify.
            code_register (QuantumRegister): The quantum register containing the state qubits.
            ancilla_qubits (QuantumRegister): The quantum register containing the ancilla qubits.
            ancilla_classical_register (ClassicalRegister): The classical register for syndrome measurement.
        """
        qc.reset(ancilla_register)
        qc.h(ancilla_register)
        for i, stabilizer in enumerate(self.stabilizer_generators):
            for j, pauli_op in enumerate(stabilizer):
                if isinstance(pauli_op, IGate):
                    continue
                qc.append(pauli_op.control(1), [ancilla_register[i], code_register[j]])
        qc.h(ancilla_register)

        qc.barrier()
        qc.measure(ancilla_register, ancilla_classical_register)

        # Implement classically conditioned corrections
        correction_gate = ZGate() if self.phase_flip else XGate()
        for i in range(self.code_length):
            with qc.if_test((ancilla_classical_register, i + 1)):
                qc.append(correction_gate, [code_register[i]])

        qc.barrier()

        return qc

    @abstractmethod
    def _encode_subroutine(self):
        """Encode the given state qubits into the encoded state.

        Returns:
            QuantumCircuit: The encoding circuit.
        """
        raise NotImplementedError

    def encoding_circuit(self, qc, state_qubits):
        """Modify the given quantum circuit by adding the encoding circuit.

        Args:
            qc (QuantumCircuit): The quantum circuit to modify.
            state_qubits (QuantumRegister): The quantum register containing the state qubits.
        """
        if len(state_qubits) != self.code_length:
            raise ValueError(
                f"Number of state qubits ({len(state_qubits)}) must match code length ({self.code_length})."
            )
        qc.compose(self._encode_subroutine(), qubits=state_qubits, inplace=True)
        return qc

    # TODO: QuantumCircuit.inverse() will not work for QutritUnitary gates
    # this is because rather than just playing the gate in reverse, it
    #  actually computes the adjoints...
    def decoding_circuit(self, qc, state_qubits):
        """Modify the given quantum circuit by adding the decoding circuit.

        Args:
            qc (QuantumCircuit): The quantum circuit to modify.
            state_qubits (QuantumRegister): The quantum register containing the encoded state qubits.
        """
        if len(state_qubits) != self.code_length:
            raise ValueError(
                f"Number of state qubits ({len(state_qubits)}) must match code length ({self.code_length})."
            )
        qc = qc.compose(
            self._encode_subroutine().inverse(), qubits=state_qubits, inplace=True
        )
        return qc


class RepetitionEncoding(QuantumErrorCorrectionCode):
    """Repetition code encoding, decoding, and correction of quantum states.

    This can implement both bit-flip and phase-flip repetition codes.
    """

    code_length = 3
    num_ancilla = 2

    def __init__(self, phase_flip=False):
        """Initialize the encoding class with specified code length and type.

        Args:
            phase_flip (bool): If True, use phase-flip code (+++, ---); otherwise, use bit-flip code (000, 111).
        """
        self.phase_flip = phase_flip
        self.stabilizer_generators = []

        # Define encoded states based on the type of code
        if phase_flip:
            encoded0 = tensor([basis(2, 0) + basis(2, 1)] * self.code_length)  # |+++>
            encoded1 = tensor([basis(2, 0) - basis(2, 1)] * self.code_length)  # |--->

            self.stabilizer_generators.append([XGate(), XGate(), IGate()])
            self.stabilizer_generators.append([IGate(), XGate(), XGate()])

        else:
            encoded0 = tensor([basis(2, 0)] * self.code_length)  # |000>
            encoded1 = tensor([basis(2, 1)] * self.code_length)  # |111>

            self.stabilizer_generators.append([ZGate(), ZGate(), IGate()])
            self.stabilizer_generators.append([IGate(), ZGate(), ZGate()])

        encoded_zero = encoded0.unit()
        encoded_one = encoded1.unit()

        super().__init__(encoded_zero, encoded_one)

    def _encode_subroutine(self):
        temp = QuantumCircuit(self.code_length)

        # Entangle qubits using CNOT gates
        for i in range(1, self.code_length):
            temp.cx(0, i)

        # For phase-flip code, apply Hadamard gates to each qubit
        if self.phase_flip:
            for i in range(self.code_length):
                temp.h(i)

        return temp


class QutritPhaseRepetitionCode(QuantumErrorCorrectionCode):
    """SNAIL concatenated erasure encoding with error ancillas.

    The logical basis is defined as:
        |L_0> = |g+f, g+f, g+f>
        |L_1> = |g-f, g-f, g-f>
    """

    code_length = 3
    num_ancilla = 2
    phase_flip = True

    def __init__(self):
        """Initialize a concatenated SNAIL encoding."""
        logical0 = tensor(p_gf, p_gf, p_gf).unit()
        logical1 = tensor(m_gf, m_gf, m_gf).unit()

        self.stabilizer_generators = []
        self.stabilizer_generators.append([x_gf, x_gf, IGate()])
        self.stabilizer_generators.append([IGate(), x_gf, x_gf])

        super().__init__(logical0, logical1)

    def _encode_subroutine(self):
        temp = QuantumCircuit(self.code_length)

        # Entangle qubits using CNOT gates
        for i in range(1, self.code_length):
            temp.append(cx_gf, [0, i])

        # For phase-flip code, apply Hadamard gates to each qubit
        for i in range(self.code_length):
            temp.append(h_gf, [i])

        return temp


# class PhaseReptition(LogicalEncoding):
#     """3-bit phase repetition encoding.

#     |L_0> = |+++> |L_1> = |--->
#     """

#     def __init__(self):
#         """Initialize a 3-bit phase repetition encoding."""
#         logical0 = tensor(p_ge, p_ge, p_ge)
#         logical1 = tensor(m_ge, m_ge, m_ge)
#         super().__init__(logical0, logical1)


# class DualRail(LogicalEncoding):
#     """Dual-rail encoding.

#     The logical basis is defined as:
#         |L_0> = |01>
#         |L_1> = |10>

#     Ancilla:
#         - |e><00| for detecting photon loss
#         - Raise DetectionError (not correctable)

#     Error Channels:
#         - Amplitude damping
#     """

#     def __init__(self):
#         """Initialize a dual-rail encoding."""
#         logical0 = tensor(basis(2, 0), basis(2, 1))
#         logical1 = tensor(basis(2, 1), basis(2, 0))
#         # to protect against photon loss, define error state as |00>
#         loss_state = tensor(basis(2, 0), basis(2, 0))
#         double_state = tensor(basis(2, 1), basis(2, 1))
#         logical_basis = LogicalBasis(logical0, logical1)
#         ancilla = Ancilla(loss_state, double_state)
#         super().__init__(logical_basis, ancilla)


# class VSLQ(LogicalEncoding):
#     """Very small logical qubit encoding.

#     The logical basis is defined as:
#         |L_0> = |g+f, g+f>
#         |L_1> = |g-f, g-f>

#     Protects against a single photon loss.
#     Reference: https://arxiv.org/pdf/1510.06117.pdf
#     """

#     def __init__(self):
#         """Initialize a VSLQ encoding."""
#         logical0 = tensor(p_gf, p_gf)
#         logical1 = tensor(m_gf, m_gf)
#         logical_basis = LogicalBasis(logical0, logical1)
#         super().__init__(logical_basis)


# class StarCode(LogicalEncoding):
#     """Autonomous quantum error correction with a star code.

#     The logical basis is defined as:
#         |L_0> = |gf> - |fg> = |g-f, f-g>
#         |L_1> = |gg> - |ff> = |g-f, g-f>

#     Reference: https://arxiv.org/pdf/2302.06707.pdf

#     Key idea in this paper is that we can use |ee> as intermediate state,
#     so can use only 2-photon drives rather than the 4-photon drives.
#     """

#     def __init__(self):
#         """Initialize a star code encoding."""
#         logical0 = tensor(p_gf, m_gf) - tensor(m_gf, p_gf)
#         logical1 = tensor(p_gf, p_gf) - tensor(m_gf, m_gf)
#         logical_basis = LogicalBasis(logical0, logical1)
#         super().__init__(logical_basis)


# class SNAILConcatWithAncilla(LogicalEncoding):
#     """SNAIL concatenated erasure encoding with error ancillas.

#     The logical basis is defined as:
#         |L_0> = |g+f, g+f, g+f>
#         |L_1> = |g-f, g-f, g-f>

#     Includes ancillas for detecting specific errors.
#     """

#     def __init__(self):
#         """Initialize a concatenated SNAIL encoding."""
#         logical0 = tensor(p_gf, p_gf, p_gf)
#         logical1 = tensor(m_gf, m_gf, m_gf)
#         logical_basis = LogicalBasis(logical0, logical1)

#         # Define ancillas for error detection and correction
#         Ancilla(
#             tensor(e, p_gf, p_gf),
#             tensor(p_gf, e, p_gf),
#             tensor(p_gf, p_gf, e),
#             tensor(e, m_gf, m_gf),
#             tensor(m_gf, e, m_gf),
#             tensor(m_gf, m_gf, e),
#         )
#         phase_ancilla_1 = Ancilla(
#             tensor(p_gf, m_gf, p_gf),
#             tensor(p_gf, m_gf, m_gf),
#             tensor(m_gf, p_gf, p_gf),
#             tensor(m_gf, p_gf, m_gf),
#         )

#         phase_ancilla_2 = Ancilla(
#             tensor(p_gf, p_gf, m_gf),
#             tensor(p_gf, m_gf, p_gf),
#             tensor(m_gf, p_gf, m_gf),
#             tensor(m_gf, m_gf, p_gf),
#         )

#         super().__init__(
#             logical_basis, phase_ancilla_1, phase_ancilla_2
#         )  # , erasure_ancilla)
