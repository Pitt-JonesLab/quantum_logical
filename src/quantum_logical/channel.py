"""Error channels for quantum systems."""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from qutip import Qobj, qeye, tensor

__all__ = ["AmplitudeDamping", "PhaseDamping", "Channel"]


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

    def __init__(self, num_qubits, hilbert_space_dim, **kwargs):
        """Initialize the channel with specified dimensions."""
        super().__init__(dims=hilbert_space_dim**num_qubits)
        self.num_qubits = num_qubits
        self.hilbert_space_dim = hilbert_space_dim
        self.params = kwargs
        self._trotter_dt = None
        self._E = None

        # Each param in params should either be a single value or an iterable
        # If iterable, then it should have length equal to num_qubits
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                self.params[key] = [value] * num_qubits
            elif len(value) != num_qubits:
                raise ValueError(
                    f"Number of parameters for {key} must equal number of qubits."
                )

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
        qubit_operators = []

        # Iterate over all qubits
        for qubit in range(self.num_qubits):
            qubit_params = {key: value[qubit] for key, value in self.params.items()}
            qubit_operators.append(self._create_single_qubit_operators(**qubit_params))
        self._E = self._extend_kraus_operators_to_multiple(qubit_operators)
        self._E = [np.array(E.full(), dtype=complex) for E in self._E]
        self._verify_completeness()
        return self._E

    def _extend_kraus_operators_to_multiple(self, qubit_operators):
        """Extend single-qubit Kraus operators to multiple qubits.

        This method assumes that errors occur independently on each
        qubit. It creates a new set of Kraus operators where the
        original single-qubit operators are applied independently to
        each qubit, while the identity operator is applied to the other
        qubits. This approach effectively models the situation where an
        error can occur on any one of the qubits, but simultaneous
        errors on multiple qubits (higher-order errors) are not
        explicitly modeled, which is a common assumption in many quantum
        error correction scenarios.

        Args:
            qubit_operators (list): List of single-qubit Kraus operators, where each
            sublist contains the Kraus operators for a specific qubit.

        Returns:
            list: List of extended Kraus operators for the full system.
        """
        if self.num_qubits == 1:
            return qubit_operators[0]

        identity = qeye(self.hilbert_space_dim)
        extended_operators = []

        for qubit in range(self.num_qubits):
            # Get the Kraus operators for the current qubit
            single_qubit_operators = qubit_operators[qubit]

            for op in single_qubit_operators:
                # Create a list of identity operators for all qubits
                operators_on_all_qubits = [identity for _ in range(self.num_qubits)]

                # Replace the identity operator with the actual Kraus operator for the current qubit
                operators_on_all_qubits[qubit] = Qobj(op)

                # Tensor the operators together
                extended_operator = tensor(*operators_on_all_qubits)
                extended_operators.append(extended_operator)

        # Apply normalization
        normalization_factor = np.sqrt(self.num_qubits)
        extended_operators = [op / normalization_factor for op in extended_operators]

        return extended_operators

    @abstractmethod
    def _create_single_qubit_operators(self, **kwargs):
        """Create and return single-qubit Kraus operators.

        This method should be implemented by each subclass.
        """
        pass


class AmplitudeDamping(Channel):
    """Amplitude damping channel for qubits."""

    def __init__(
        self,
        T1: Union[float, List[float]],
        num_qubits: int = 1,
        hilbert_space_dim: int = 2,
    ):
        """Initialize the amplitude damping channel.

        Args:
            T1 (float or list of floats): The T1 energy relaxation rate for the qubits.
            num_qubits (int): The number of qubits on which the channel acts.
            hilbert_space_dim (int): The dimension of the Hilbert space of each qubit.
        """
        super().__init__(num_qubits, hilbert_space_dim, T1=T1)

    def _create_single_qubit_operators(self, T1):
        """Create single-qubit Kraus operators for amplitude damping."""
        if self.hilbert_space_dim == 2:  # standard qubit case
            _gamma = 1 - np.exp(-self._trotter_dt / T1)
            E0_single = np.array([[1, 0], [0, np.sqrt(1 - _gamma)]])
            E1_single = np.array([[0, np.sqrt(_gamma)], [0, 0]])
            return [E0_single, E1_single]

        elif self.hilbert_space_dim == 3:  # qutrit case
            # Simplified qutrit damping model
            # Ref: M. Grassl, et al. doi: 10.1109/TIT.2018.2790423.
            # assuming f->e and e->g are the same and f->g is 0

            # Simplified transition rates
            _gamma_fe = _gamma_eg = 1 - np.exp(-self._trotter_dt / T1)
            _gamma_fg = 0  # Neglected direct transition from f to g

            # Simplified Kraus operators
            A_01 = np.sqrt(_gamma_eg) * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
            A_12 = np.sqrt(_gamma_fe) * np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
            A_02 = np.sqrt(_gamma_fg) * np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
            A_0 = (
                np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
                + np.sqrt(1 - _gamma_eg) * np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
                + np.sqrt(1 - _gamma_fg - _gamma_fe)
                * np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
            )

            return [A_0, A_01, A_12, A_02]

        else:
            raise NotImplementedError("Unsupported Hilbert space dimension.")


class PhaseDamping(Channel):
    """Phase damping channel for qubits."""

    def __init__(
        self,
        T1: Union[float, List[float]],
        T2: Union[float, List[float]],
        num_qubits: int = 1,
        hilbert_space_dim: int = 2,
    ):
        """Initialize the phase damping channel.

        Args:
            T1 (float or list of floats): The T1 energy relaxation rate for the qubits.
            T2 (float or list of floats): The T2 dephrasing rate for the qubits.
            num_qubits (int): The number of qubits on which the channel acts.
            hilbert_space_dim (int): The dimension of the Hilbert space of each qubit.
        """
        Tphi = self._Tphi(T1, T2)
        super().__init__(num_qubits, hilbert_space_dim, Tphi=Tphi)

    @staticmethod
    def _Tphi(T1, T2):
        """Calculate the dephasing time T_phi from T1 and T2."""
        if isinstance(T1, list) and isinstance(T2, list):
            return [1 / (1 / t2 - 1 / (2 * t1)) for t1, t2 in zip(T1, T2)]
        else:
            return 1 / (1 / T2 - 1 / (2 * T1))

    def _create_single_qubit_operators(self, Tphi):
        """Create single-qubit Kraus operators for phase damping."""
        if self.hilbert_space_dim == 2:  # standard qubit case
            _gamma = 1 - np.exp(-self._trotter_dt / Tphi)
            E0_single = np.array([[1, 0], [0, np.sqrt(1 - _gamma)]])
            E1_single = np.array([[0, 0], [0, np.sqrt(_gamma)]])
            return [E0_single, E1_single]

        elif self.hilbert_space_dim == 3:  # qutrit case
            _gamma = 1 - np.exp(-self._trotter_dt / Tphi)
            E0_single = np.array(
                [[1, 0, 0], [0, np.sqrt(1 - _gamma), 0], [0, 0, np.sqrt(1 - _gamma)]]
            )
            E1_single = np.array([[0, 0, 0], [0, np.sqrt(_gamma), 0], [0, 0, 0]])
            E2_single = np.array([[0, 0, 0], [0, 0, 0], [0, 0, np.sqrt(_gamma)]])
            return [E0_single, E1_single, E2_single]

        else:
            raise NotImplementedError("Unsupported Hilbert space dimension.")
