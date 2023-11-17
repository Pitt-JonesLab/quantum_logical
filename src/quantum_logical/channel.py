"""Error channels for quantum systems."""
from abc import ABC, abstractmethod

import numpy as np
from qutip import Qobj, qeye, tensor


class CPTPMap(ABC):
    """Abstract class for operators, unitary gates or error channels.

    Operators are callable objects that act on quantum states.
    """

    def __init__(self, dims):
        """Initialize the CPTPMap with a specified dimension."""
        self.dims = dims

    def __call__(self, state):
        """Apply the CPTP map to a given quantum state."""
        if self._E is None:
            raise ValueError("Operators have not been initialized.")
        state_numpy = state.full() if isinstance(state, Qobj) else state
        return sum([E @ state_numpy @ E.T.conj() for E in self.E])

    def _verify_completeness(self):
        """Verify that Kraus operators satisfy the completeness relation."""
        completeness = sum([E.conj().T @ E for E in self.E])
        assert np.allclose(
            completeness, np.eye(self.dims), atol=1e-6
        ), "Kraus operators do not satisfy the completeness relation"

    @abstractmethod
    def _init_kraus_operators(self):
        """Initialize and return Kraus operators.

        This method should be implemented by each subclass.
        """
        pass


class Channel(CPTPMap):
    """Base class for quantum error channels."""

    def __init__(self, dims):
        """Initialize the channel with specified dimensions."""
        super().__init__(dims)
        self._trotter_dt = None
        self._E = None

    @property
    def E(self):
        """Return the Kraus operators for the channel."""
        if self._E is None:
            raise ValueError("Kraus operators have not been initialized.")
        return self._E

    def set_trotter_dt(self, trotter_dt):
        """Set the trotter step size and initialize Kraus operators."""
        self._trotter_dt = trotter_dt
        self._E = self._init_kraus_operators()
        return True

    def _init_kraus_operators(self):
        """Initialize and extend Kraus operators to multiple qubits."""
        single_qubit_operators = self._create_single_qubit_operators()
        self._E = self._extend_kraus_operators_to_multiple_qubits(
            single_qubit_operators
        )
        self._E = [np.array(E, dtype=complex) for E in self._E]
        self._verify_completeness()
        return self._E

    def _extend_kraus_operators_to_multiple_qubits(self, single_qubit_operators):
        """Extend single-qubit Kraus operators to multiple qubits."""
        if self.num_qubits == 1:
            return single_qubit_operators

        identity = qeye(2)
        extended_operators = []
        for qubit in range(self.num_qubits):
            for op in single_qubit_operators:
                operators = [
                    Qobj(op) if i == qubit else identity for i in range(self.num_qubits)
                ]
                extended_operators.append(tensor(*operators))

        # Apply normalization
        normalization_factor = np.sqrt(self.num_qubits)
        extended_operators = [op / normalization_factor for op in extended_operators]

        return extended_operators

    @abstractmethod
    def _create_single_qubit_operators(self):
        """Create and return single-qubit Kraus operators.

        This method should be implemented by each subclass.
        """
        pass


class AmplitudeDamping(Channel):
    """Amplitude damping channel for qubits."""

    def __init__(self, T1, num_qubits=1):
        """Initialize with a given T1 relaxation time and number of qubits."""
        self.T1 = T1
        self.num_qubits = num_qubits
        super().__init__(dims=2**num_qubits)

    def _create_single_qubit_operators(self):
        """Create single-qubit Kraus operators for amplitude damping."""
        _gamma = 1 - np.exp(-self._trotter_dt / self.T1)
        E0_single = np.array([[1, 0], [0, np.sqrt(1 - _gamma)]])
        E1_single = np.array([[0, np.sqrt(_gamma)], [0, 0]])
        return [E0_single, E1_single]


class PhaseDamping(Channel):
    """Phase damping channel for qubits."""

    def __init__(self, T2, num_qubits=1):
        """Initialize with a given T2 dephasing time and number of qubits."""
        self.T2 = T2
        self.num_qubits = num_qubits
        super().__init__(dims=2**num_qubits)

    def _create_single_qubit_operators(self):
        """Create single-qubit Kraus operators for phase damping."""
        _lambda = 1 - np.exp(-self._trotter_dt / self.T2)
        E0_single = np.array([[1, 0], [0, np.sqrt(1 - _lambda)]])
        E1_single = np.array([[0, 0], [0, np.sqrt(_lambda)]])
        return [E0_single, E1_single]

    # @classmethod
    # def extend_to_multiple_qubits(cls, original_channel, num_qubits):
    #     """
    #     Extend a single-qubit Amplitude Damping channel to multiple qubits.

    #     This method assumes that errors occur independently on each qubit.
    #     It creates a new set of Kraus operators where the original single-qubit
    #     operators are applied independently to each qubit, while the identity
    #     operator is applied to the other qubits. This approach effectively
    #     models the situation where an error can occur on any one of the qubits,
    #     but simultaneous errors on multiple qubits (higher-order errors) are
    #     not explicitly modeled, which is a common assumption in many quantum
    #     error correction scenarios.

    #     Args:
    #         original_channel (AmplitudeDamping): The original single-qubit amplitude damping channel.
    #         num_qubits (int): The number of qubits to extend the channel to.

    #     Returns:
    #         AmplitudeDamping: A new Amplitude Damping channel instance extended to multiple qubits.
    #     """
    #     if original_channel.dims != 2:
    #         raise ValueError("Original channel must be a single-qubit channel.")

    #     # Create a new instance using the specific subclass constructor
    #     new_channel = cls(**original_channel.__dict__)

    #     # Reset the Kraus operators and set new dimensions
    #     new_channel._E = None
    #     new_channel.dims = 2**num_qubits

    #     # Generate extended operators
    #     extended_operators = []
    #     identity = np.eye(2)
    #     for op in original_channel.E:
    #         for i in range(num_qubits):
    #             operators = [op if k == i else identity for k in range(num_qubits)]
    #             extended_operators.append(tensor(*operators))

    #     new_channel._E = extended_operators
    #     return new_channel


# def _get_fractional_unitary(self, unitary, num_steps):
#     """Compute and cache the fractional unitary."""
#     if unitary is not None and num_steps > 0:
#         key = (unitary.tobytes(), num_steps)  # Unique key for caching
#         if key not in self.cached_fractional_unitaries:
#             fractional_unitary = fractional_matrix_power(unitary, 1 / num_steps)
#             self.cached_fractional_unitaries[key] = fractional_unitary
#         return self.cached_fractional_unitaries[key]
#     return None

# def _apply(self, state_numpy, num_steps, operators):
#     """Apply a sequence of operators to a given quantum state using
#     numpy."""
#     for _ in range(num_steps):
#         for op_set in operators:
#             if isinstance(op_set, list):  # This is a set of Kraus operators
#                 new_state = np.zeros_like(state_numpy)
#                 for op in op_set:
#                     new_state += op @ state_numpy @ op.T.conj()
#                 state_numpy = new_state
#             else:  # This is the unitary
#                 state_numpy = op_set @ state_numpy @ op_set.T.conj()
#     return state_numpy

# def apply_error_channel(self, state, duration, unitary=None, use_rust=0):
#     num_steps = int(duration / self.trotter_dt)

#     # Convert Qobj to numpy array for calculation
#     state_numpy = state.full() if isinstance(state, Qobj) else state
#     unitary = unitary.full() if isinstance(unitary, Qobj) else unitary

#     if num_steps == 0:
#         return (
#             Qobj(unitary @ state_numpy @ unitary.T.conj(), dims=state.dims)
#             if unitary is not None
#             else state
#         )

#     operators = [self.E.copy()]  # Nested list for Kraus operators
#     fractional_unitary = self._get_fractional_unitary(unitary, num_steps)
#     if fractional_unitary is not None:
#         operators.append(fractional_unitary)  # Unitary is a separate element

#     if use_rust:
#         raise NotImplementedError("Rust apply needs debug")
#         # Update this call to match the new Rust function signature
#         state_numpy = rust_apply(state_numpy, num_steps, operators)
#     else:
#         state_numpy = self._apply(state_numpy, num_steps, operators)

#     return Qobj(state_numpy, dims=state.dims) / np.trace(state_numpy)


# class MultiQubitExtensionMixin:
#     # def extend_to_qubits(self, num_qubits):
#     #     """Extend the channel to act identically on multiple qubits."""
#     #     extended_ops = []
#     #     for op in self.E:
#     #         for i in range(num_qubits):
#     #             extended_op = [op if k == i else qeye(2) for k in range(num_qubits)]
#     #             extended_ops.append(tensor(*extended_op))
#     #     return extended_ops


# class QutritAmplitudeDamping(ErrorChannel):
#     """Amplitude damping channel for a qutrit."""

#     def __init__(self, tau, k1, k2, trotter_dt):
#         """Initialize with a given tau, k1, k2, and trotter step size."""
#         self.tau = tau
#         self.k1 = k1
#         self.k2 = k2
#         super().__init__(trotter_dt, dims=3)

#     def _init_kraus_operators(self):
#         zero_ket = np.array([[1], [0], [0]])
#         one_ket = np.array([[0], [1], [0]])
#         two_ket = np.array([[0], [0], [1]])

#         gamma_01 = 2 * self.k2 * self.tau
#         gamma_02 = 2 * self.k1 * self.k2 * (self.tau) ** 2
#         gamma_12 = 2 * self.k1 * self.tau

#         A_01 = np.sqrt(gamma_01) * zero_ket @ one_ket.T.conj()
#         A_12 = np.sqrt(gamma_12) * one_ket @ two_ket.T.conj()
#         A_02 = np.sqrt(gamma_02) * zero_ket @ two_ket.T.conj()
#         A_0 = (
#             zero_ket @ zero_ket.T.conj()
#             + np.sqrt(1 - gamma_01) * one_ket @ one_ket.T.conj()
#             + np.sqrt(1 - gamma_02 - gamma_12) * two_ket @ two_ket.T.conj()
#         )

#         return [A_0, A_01, A_12, A_02]


# class MultiQubitErrorChannel(ErrorChannel):
#     """A class for error channels that can be applied to multi-qubit
#     systems."""

#     def __init__(self, N, T1s, T2s, trotter_dt):
#         """Initializes with given arrays of T1 and T2 times for each qubit, and
#         a trotter step size."""
#         self.N = N
#         assert len(T1s) == len(
#             T2s
#         ), "Arrays of T1 and T2 times must have the same length"
#         self.num_qubits = len(T1s)
#         self.T1s = T1s
#         self.T2s = T2s
#         dims = 2**self.num_qubits
#         self.trotter_dt = trotter_dt
#         super().__init__(trotter_dt, dims)

#     def _init_kraus_operators(self):
#         """Creates the multi-qubit Kraus operators for the error channels."""
#         # Create single-qubit Kraus operators for amplitude and phase damping for each qubit
#         kraus_ops = []

#         for T1, T2 in zip(self.T1s, self.T2s):
#             # Combine amplitude and phase damping operators for each qubit
#             amps = AmplitudeDamping(T1, self.trotter_dt).E
#             phases = PhaseDamping(T2, self.trotter_dt).E
#             combined_ops = amps + phases

#             # Normalize the combined operators for each qubit
#             normalization_factor = np.sqrt(2 * self.N)  # ?
#             normalized_ops = [Qobj(op) / normalization_factor for op in combined_ops]

#             kraus_ops.append(normalized_ops)

#         # Tensor the single-qubit Kraus operators together with identity operators for the other qubits
#         multi_qubit_kraus_ops = []

#         for i in range(self.num_qubits):
#             for op in kraus_ops[i]:
#                 kraus_op_multi = [
#                     op if k == i else qeye(2) for k in range(self.num_qubits)
#                 ]
#                 multi_qubit_kraus_ops.append(tensor(*kraus_op_multi))

#         return multi_qubit_kraus_ops


# class MultiQutritErrorChannel(ErrorChannel):
#     N = 2

#     def __init__(self, tau, k1, k2, trotter_dt, num_qutrits):
#         self.num_qutrits = num_qutrits
#         self.tau = tau
#         self.k1 = k1
#         self.k2 = k2
#         self.trotter_dt = trotter_dt
#         dims = 3**num_qutrits
#         super().__init__(trotter_dt, dims)

#     def _init_kraus_operators(self):
#         kraus_ops = []
#         for _ in range(self.num_qutrits):
#             # Initialize the single-qutrit amplitude damping channel
#             single_qutrit_channel = QutritAmplitudeDamping(
#                 self.tau, self.k1, self.k2, self.trotter_dt
#             )
#             kraus_ops_single = single_qutrit_channel.E

#             # Normalize if needed
#             normalization_factor = np.sqrt(1 * self.N)
#             normalized_ops = [
#                 Qobj(op) / normalization_factor for op in kraus_ops_single
#             ]

#             kraus_ops.append(normalized_ops)

#         # tensor wiith identity operators to create the multi-qutrit kraus operators
#         multi_qutrit_kraus_ops = []
#         for i in range(self.num_qutrits):
#             for op in kraus_ops[i]:
#                 kraus_op_multi = [
#                     op if k == i else qeye(3) for k in range(self.num_qutrits)
#                 ]
#                 multi_qutrit_kraus_ops.append(tensor(*kraus_op_multi))

#         return multi_qutrit_kraus_ops


# # TODO?
# class QutritPhaseDamping(ErrorChannel):
#     def __init__(self, lambdas, trotter_dt):
#         super().__init__(trotter_dt)
#         # Define the Kraus operators for qutrit phase damping
#         raise NotImplementedError
