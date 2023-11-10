# Ampltiude Damping Channel

# TODO implemented Generalized Amplitude Damping Channel
# this considers "hot"-qubits, ground state is not perfectly |0>

from abc import ABC

import numpy as np
from qutip import Qobj, basis


class ErrorChannel(ABC):
    """A base class for error channels in a quantum system."""

    def __init__(self, trotter_dt, dims):
        """Initializes with a given trotter step size and dimensionality."""
        self.trotter_dt = trotter_dt
        self.dims = dims
        try:
            self._verify_completeness()
        except AttributeError:
            AttributeError("Kraus operators do not satisfy the completeness relation")

    def _verify_completeness(self):
        """Verifies that the set of Kraus operators satisfies the completeness
        relation."""
        assert np.allclose(
            sum([E.dag() * E for E in self.E]),
            Qobj(np.eye(self.dims)),
            atol=1e-6,
        ), "Kraus operators do not satisfy the completeness relation"

    def _apply(self, state):
        """Applies the error channel to a given quantum state.

        Parameters:
        state (Qobj): The quantum state to which the error channel is applied.

        Returns:
        Qobj: The quantum state after applying the error channel.
        """
        return sum([E * state * E.dag() for E in self.E])

    def apply_error_channel(self, state, duration):
        """Applies the error channel to a quantum state over a specified
        duration with a given trotter step size.

        Parameters:
        state (Qobj): The quantum state to which the error channel is to be applied.
        duration (float): The total time duration for the error to be applied.

        Returns:
        Qobj: The quantum state after applying the error channel over the specified duration.
        """
        # Calculate the number of trotter steps
        num_steps = int(duration / self.trotter_dt)

        for _ in range(num_steps):
            state = self._apply(state)
        return state


# TODO combine these into a single amplitude and phase damping class,
# where dimensionality is passed in as an argument


class AmplitudeDamping(ErrorChannel):
    def __init__(self, T1, trotter_dt):
        _gamma = 1 - np.exp(-trotter_dt / T1)
        E0 = Qobj([[1, 0], [0, np.sqrt(1 - _gamma)]])
        E1 = Qobj([[0, np.sqrt(_gamma)], [0, 0]])
        self.E = [E0, E1]
        super().__init__(trotter_dt, dims=2)


class PhaseDamping(ErrorChannel):
    def __init__(self, T2, trotter_dt):
        _lambda = 1 - np.exp(-trotter_dt / T2)
        E0 = Qobj([[1, 0], [0, np.sqrt(1 - _lambda)]])
        E1 = Qobj([[0, 0], [0, np.sqrt(_lambda)]])
        self.E = [E0, E1]
        super().__init__(trotter_dt, dims=2)


class QutritAmplitudeDamping(ErrorChannel):
    # FIXME, what is tau?
    def __init__(self, tau, k1, k2, trotter_dt):
        zero_ket = basis(3, 0)
        one_ket = basis(3, 1)
        two_ket = basis(3, 2)

        gamma_01 = 2 * k2 * tau
        gamma_02 = 2 * k1 * k2 * (tau) ** 2
        gamma_12 = 2 * k1 * tau

        # Define the Kraus operators for qutrit amplitude damping
        A_01 = np.sqrt(gamma_01) * zero_ket * one_ket.dag()
        A_12 = np.sqrt(gamma_12) * one_ket * two_ket.dag()
        A_02 = np.sqrt(gamma_02) * zero_ket * two_ket.dag()
        A_0 = (
            zero_ket * zero_ket.dag()
            + np.sqrt(1 - gamma_01) * one_ket * one_ket.dag()
            + np.sqrt(1 - gamma_02 - gamma_12) * two_ket * two_ket.dag()
        )

        self.E = [A_0, A_01, A_12, A_02]
        super().__init__(trotter_dt, dims=3)


# TODO?
class QutritPhaseDamping(ErrorChannel):
    def __init__(self, lambdas, trotter_dt):
        super().__init__(trotter_dt)
        # Define the Kraus operators for qutrit phase damping
        raise NotImplementedError
