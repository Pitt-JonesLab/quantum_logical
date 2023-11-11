"""Error channels for quantum systems."""
# Ampltiude Damping Channel

# TODO implemented Generalized Amplitude Damping Channel
# this considers "hot"-qubits, ground state is not perfectly |0>

from abc import ABC, abstractmethod

import numpy as np
from qutip import Qobj

from quantum_logical._lib import apply_error_channel_rust

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

    def _apply(self, state_numpy, num_steps):
        """Apply the error channel to a given quantum state using numpy."""
        for _ in range(num_steps):
            state_numpy = sum([E @ state_numpy @ E.T.conj() for E in self.E])
        return state_numpy

    def apply_error_channel(self, state, duration, use_rust=True):
        """Apply the channel to a state over a specified duration."""
        # check if divides duration into integer number of steps
        # otherwise, will not apply channel for proper duration
        # if not np.isclose(duration % self.trotter_dt, 0):
        #     error = duration % self.trotter_dt
        #     tolerance = 1e-6
        #     if error > tolerance:
        #         warnings.warn(
        #             f"Duration is not a multiple of trotter_dt. Error: {error}.",
        #             UserWarning,
        #         )

        num_steps = int(duration / self.trotter_dt)
        state_numpy = state.full()  # Convert Qobj to numpy array for calculation
        # assert all kraus operators are numpy arrays
        assert all([isinstance(E, np.ndarray) for E in self.E])

        if use_rust:
            state_numpy = apply_error_channel_rust(state_numpy, num_steps, self.E)
        else:
            state_numpy = self._apply(state_numpy, num_steps)

        # Convert the final numpy array back to Qobj
        # preserve the dimensions of the original state
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


# # TODO?
# class QutritPhaseDamping(ErrorChannel):
#     def __init__(self, lambdas, trotter_dt):
#         super().__init__(trotter_dt)
#         # Define the Kraus operators for qutrit phase damping
#         raise NotImplementedError
