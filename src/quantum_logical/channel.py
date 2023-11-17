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

    # TODO: Make this so we could extend with varying T1 and T2 between qubits
    def _extend_kraus_operators_to_multiple_qubits(self, single_qubit_operators):
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
        """
        if self.num_qubits == 1:
            return single_qubit_operators

        identity = qeye(self.hilbert_space_dim)
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

    def __init__(self, T1, num_qubits=1, hilbert_space_dim=2):
        """Initialize with a given T1 relaxation time and number of qubits."""
        self.T1 = T1
        self.num_qubits = num_qubits
        self.hilbert_space_dim = hilbert_space_dim
        super().__init__(dims=hilbert_space_dim**num_qubits)

    def _create_single_qubit_operators(self):
        """Create single-qubit Kraus operators for amplitude damping."""
        if self.hilbert_space_dim == 2:  # standard qubit case
            _gamma = 1 - np.exp(-self._trotter_dt / self.T1)
            E0_single = np.array([[1, 0], [0, np.sqrt(1 - _gamma)]])
            E1_single = np.array([[0, np.sqrt(_gamma)], [0, 0]])
            return [E0_single, E1_single]

        elif self.hilbert_space_dim == 3:  # qutrit case
            # Simplified qutrit damping model
            # Ref: M. Grassl, et al. doi: 10.1109/TIT.2018.2790423.
            # assuming f->e and e->g are the same and f->g is 0

            # Simplified transition rates
            _gamma_fe = _gamma_eg = 1 - np.exp(-self._trotter_dt / self.T1)
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

    def __init__(self, T2, num_qubits=1, hilbert_space_dim=2):
        """Initialize with a given T2 dephasing time and number of qubits."""
        self.T2 = T2
        self.num_qubits = num_qubits
        self.hilbert_space_dim = hilbert_space_dim
        super().__init__(dims=hilbert_space_dim**num_qubits)

    def _create_single_qubit_operators(self):
        """Create single-qubit Kraus operators for phase damping."""
        if self.hilbert_space_dim == 2:  # standard qubit case
            _gamma = 1 - np.exp(-self._trotter_dt / self.T2)
            E0_single = np.array([[1, 0], [0, np.sqrt(1 - _gamma)]])
            E1_single = np.array([[0, 0], [0, np.sqrt(_gamma)]])
            return [E0_single, E1_single]

        else:
            raise NotImplementedError("Unsupported Hilbert space dimension.")
