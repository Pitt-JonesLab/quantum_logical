"""Error channels for quantum systems."""
# Ampltiude Damping Channel

# TODO implemented Generalized Amplitude Damping Channel
# this considers "hot"-qubits, ground state is not perfectly |0>

from abc import ABC, abstractmethod

import numpy as np
from qutip import Qobj, qeye, tensor
from scipy.linalg import fractional_matrix_power

from quantum_logical._lib import apply_operators_in_place as rust_apply

# import warnings


class ErrorChannel(ABC):
    """A base class for error channels in a quantum system."""

    def __init__(self, trotter_dt, dims):
        """Initialize with a given trotter step size."""
        self.trotter_dt = trotter_dt
        self.dims = dims
        self.E = self._init_kraus_operators()
        # convert E to complex numpy arrays
        self.E = [np.array(E, dtype=complex) for E in self.E]
        self._verify_completeness()
        self.cached_fractional_unitaries = {}

    @abstractmethod
    def _init_kraus_operators(self):
        """Initialize and return Kraus operators as numpy arrays."""
        pass

    def _verify_completeness(self):
        """Verify Kraus operators satisfies the completeness relation."""
        completeness = sum([E.conj().T @ E for E in self.E])
        assert np.allclose(
            completeness, np.eye(self.dims), atol=1e-6
        ), "Kraus operators do not satisfy the completeness relation"

    # @classmethod
    # def from_combined_channels(cls, channels, trotter_dt, dims):
    #     """Create a new ErrorChannel from a combination of other channels."""
    #     combined_kraus_operators = sum([channel.E for channel in channels], [])
    #     normalization_factor = 1 / np.sqrt(len(combined_kraus_operators))
    #     normalized_operators = [
    #         op * normalization_factor for op in combined_kraus_operators
    #     ]

    #     new_channel = cls(trotter_dt, dims)
    #     new_channel.E = normalized_operators
    #     return new_channel

    def _get_fractional_unitary(self, unitary, num_steps):
        """Compute and cache the fractional unitary."""
        if unitary is not None and num_steps > 0:
            key = (unitary.tobytes(), num_steps)  # Unique key for caching
            if key not in self.cached_fractional_unitaries:
                fractional_unitary = fractional_matrix_power(unitary, 1 / num_steps)
                self.cached_fractional_unitaries[key] = fractional_unitary
            return self.cached_fractional_unitaries[key]
        return None

    def _apply(self, state_numpy, num_steps, operators):
        """Apply a sequence of operators to a given quantum state using
        numpy."""
        for _ in range(num_steps):
            new_state = np.zeros_like(state_numpy)
            for op in operators:
                new_state += op @ state_numpy @ op.T.conj()
            state_numpy = new_state
        return state_numpy

    def apply_error_channel(self, state, duration, unitary=None, use_rust=True):
        """Apply the channel to a state over a specified duration."""
        num_steps = int(duration / self.trotter_dt)
        state_numpy = state.full()  # Convert Qobj to numpy array for calculation

        if num_steps == 0:
            return (
                state
                if unitary is None
                else Qobj(unitary @ state_numpy @ unitary.T.conj(), dims=state.dims)
            )

        operators = self.E.copy()
        fractional_unitary = self._get_fractional_unitary(unitary, num_steps)
        if fractional_unitary is not None:
            operators.insert(0, fractional_unitary)

        if use_rust:
            # Update this call to match the new Rust function signature
            state_numpy = rust_apply(state_numpy, num_steps, operators)
        else:
            state_numpy = self._apply(state_numpy, num_steps, operators)

        return Qobj(state_numpy, dims=state.dims)


class AmplitudeDamping(ErrorChannel):
    """Amplitude damping channel for a qubit."""

    def __init__(self, T1, trotter_dt):
        """Initialize with a given T1 time and trotter step size."""
        self.T1 = T1
        super().__init__(trotter_dt, dims=2)

    def _init_kraus_operators(self):
        _gamma = 1 - np.exp(-self.trotter_dt / self.T1)
        E0 = np.array([[1, 0], [0, np.sqrt(1 - _gamma)]])
        E1 = np.array([[0, np.sqrt(_gamma)], [0, 0]])
        return [E0, E1]


class PhaseDamping(ErrorChannel):
    """Phase damping channel for a qubit."""

    def __init__(self, T2, trotter_dt):
        """Initialize with a given T2 time and trotter step size."""
        self.T2 = T2
        super().__init__(trotter_dt, dims=2)

    def _init_kraus_operators(self):
        _lambda = 1 - np.exp(-self.trotter_dt / self.T2)
        E0 = np.array([[1, 0], [0, np.sqrt(1 - _lambda)]])
        E1 = np.array([[0, 0], [0, np.sqrt(_lambda)]])
        return [E0, E1]


class QutritAmplitudeDamping(ErrorChannel):
    """Amplitude damping channel for a qutrit."""

    def __init__(self, tau, k1, k2, trotter_dt):
        """Initialize with a given tau, k1, k2, and trotter step size."""
        self.tau = tau
        self.k1 = k1
        self.k2 = k2
        super().__init__(trotter_dt, dims=3)

    def _init_kraus_operators(self):
        zero_ket = np.array([[1], [0], [0]])
        one_ket = np.array([[0], [1], [0]])
        two_ket = np.array([[0], [0], [1]])

        gamma_01 = 2 * self.k2 * self.tau
        gamma_02 = 2 * self.k1 * self.k2 * (self.tau) ** 2
        gamma_12 = 2 * self.k1 * self.tau

        A_01 = np.sqrt(gamma_01) * zero_ket @ one_ket.T.conj()
        A_12 = np.sqrt(gamma_12) * one_ket @ two_ket.T.conj()
        A_02 = np.sqrt(gamma_02) * zero_ket @ two_ket.T.conj()
        A_0 = (
            zero_ket @ zero_ket.T.conj()
            + np.sqrt(1 - gamma_01) * one_ket @ one_ket.T.conj()
            + np.sqrt(1 - gamma_02 - gamma_12) * two_ket @ two_ket.T.conj()
        )

        return [A_0, A_01, A_12, A_02]


class MultiQubitErrorChannel(ErrorChannel):
    """A class for error channels that can be applied to multi-qubit
    systems."""

    def __init__(self, N, T1s, T2s, trotter_dt):
        """Initializes with given arrays of T1 and T2 times for each qubit, and
        a trotter step size."""
        self.N = N
        assert len(T1s) == len(
            T2s
        ), "Arrays of T1 and T2 times must have the same length"
        self.num_qubits = len(T1s)
        self.T1s = T1s
        self.T2s = T2s
        dims = 2**self.num_qubits
        self.trotter_dt = trotter_dt
        super().__init__(trotter_dt, dims)

    def _init_kraus_operators(self):
        """Creates the multi-qubit Kraus operators for the error channels."""
        # Create single-qubit Kraus operators for amplitude and phase damping for each qubit
        kraus_ops = []

        for T1, T2 in zip(self.T1s, self.T2s):
            # Combine amplitude and phase damping operators for each qubit
            amps = AmplitudeDamping(T1, self.trotter_dt).E
            phases = PhaseDamping(T2, self.trotter_dt).E
            combined_ops = amps + phases

            # Normalize the combined operators for each qubit
            normalization_factor = np.sqrt(2 * self.N)  # ?
            normalized_ops = [Qobj(op) / normalization_factor for op in combined_ops]

            kraus_ops.append(normalized_ops)

        # Tensor the single-qubit Kraus operators together with identity operators for the other qubits
        multi_qubit_kraus_ops = []

        for i in range(self.num_qubits):
            for op in kraus_ops[i]:
                kraus_op_multi = [
                    op if k == i else qeye(2) for k in range(self.num_qubits)
                ]
                multi_qubit_kraus_ops.append(tensor(*kraus_op_multi))

        return multi_qubit_kraus_ops


class MultiQutritErrorChannel(ErrorChannel):
    N = 2

    def __init__(self, tau, k1, k2, trotter_dt, num_qutrits):
        self.num_qutrits = num_qutrits
        self.tau = tau
        self.k1 = k1
        self.k2 = k2
        self.trotter_dt = trotter_dt
        dims = 3**num_qutrits
        super().__init__(trotter_dt, dims)

    def _init_kraus_operators(self):
        kraus_ops = []
        for _ in range(self.num_qutrits):
            # Initialize the single-qutrit amplitude damping channel
            single_qutrit_channel = QutritAmplitudeDamping(
                self.tau, self.k1, self.k2, self.trotter_dt
            )
            kraus_ops_single = single_qutrit_channel.E

            # Normalize if needed
            normalization_factor = np.sqrt(1 * self.N)
            normalized_ops = [
                Qobj(op) / normalization_factor for op in kraus_ops_single
            ]

            kraus_ops.append(normalized_ops)

        # tensor wiith identity operators to create the multi-qutrit kraus operators
        multi_qutrit_kraus_ops = []
        for i in range(self.num_qutrits):
            for op in kraus_ops[i]:
                kraus_op_multi = [
                    op if k == i else qeye(3) for k in range(self.num_qutrits)
                ]
                multi_qutrit_kraus_ops.append(tensor(*kraus_op_multi))

        return multi_qutrit_kraus_ops


# # TODO?
# class QutritPhaseDamping(ErrorChannel):
#     def __init__(self, lambdas, trotter_dt):
#         super().__init__(trotter_dt)
#         # Define the Kraus operators for qutrit phase damping
#         raise NotImplementedError
