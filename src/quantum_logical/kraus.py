import qutip as qt
import numpy as np
from itertools import zip_longest

class KrausOperators:
    def __init__(self, trotter_dt, T1, T2, dim, num_qubits, qudits):
        self.trotter_dt = trotter_dt
        self.T1 = T1
        self.T2 = T2
        self.dim = dim
        self.num_qubits = num_qubits
        self.qudits = qudits

    def _compute_transition_rates(self):
        """
        Compute the transition rates for the Kraus operators.
        """
        gamma = 1 - np.exp(-self.trotter_dt / self.T1)
        T_phi = (1 / (1 / self.T2 - 1 / (2 * self.T1)))  # T2 relaxation time
        gamma1 = 1 - np.exp(-self.trotter_dt / T_phi)
        return gamma, gamma1

    def _create_kraus_operators_qubit(self, gamma, gamma1):
        """
        Create Kraus operators for a qubit system.
        """
        a = qt.Qobj([[1, 0], [0, np.sqrt(1 - gamma)]])
        b = qt.Qobj([[0, np.sqrt(gamma)], [0, 0]])
        c = qt.Qobj([[1, 0], [0, np.sqrt(1 - gamma1)]])
        d = qt.Qobj([[0, 0], [0, np.sqrt(gamma1)]])
        return [a, b], [c, d]

    def _create_kraus_operators_qutrit(self, gamma, gamma1):
        """
        Create Kraus operators for a qutrit system.
        """
        _fe = _eg = 1 - np.exp(-self.trotter_dt / self.T1)
        _fg = 0  # Neglected direct transition from f to g
        A_01 = np.sqrt(_eg) * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        A_12 = np.sqrt(_fe) * np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        A_02 = np.sqrt(_fg) * np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
        A_0 = (
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            + np.sqrt(1 - _eg) * np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            + np.sqrt(1 - _fg - _fe) * np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        )
        return [qt.Qobj(A_0), qt.Qobj(A_01), qt.Qobj(A_12), qt.Qobj(A_02)]

    def _create_kraus_operators_ququart(self, gamma, gamma1):
        """
        Create Kraus operators for a ququart system.
        """
        _hf = _fe = _eg = 1 - np.exp(-self.trotter_dt / self.T1)
        _hg = _he = _fg = 0  # Neglected direct transition from f to g

        A_01 = np.sqrt(_eg) * np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        A_12 = np.sqrt(_fe) * np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        A_02 = np.sqrt(_fg) * np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        A_03 = np.sqrt(_hg) * np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        A_13 = np.sqrt(_he) * np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        A_23 = np.sqrt(_hf) * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        A_0 = (
            np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            + np.sqrt(1 - _eg) * np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            + np.sqrt(1 - _fg - _fe) * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
            + np.sqrt(1 - _hg - _he - _hf) * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        )
        return [qt.Qobj(A_0), qt.Qobj(A_01), qt.Qobj(A_12), qt.Qobj(A_02), qt.Qobj(A_03), qt.Qobj(A_13), qt.Qobj(A_23)]

    def _create_error_channels(self, kraus_ops):
        """
        Create the error channels by adding the Kraus operators with identity tensors.
        """
        errors = []
        for i in range(self.num_qubits):
            for op in kraus_ops:
                identity = [qt.qeye(self.dim) for _ in range(self.num_qubits)]
                identity[i] = op
                errors.append(1 / np.sqrt(self.num_qubits) * qt.tensor(identity))
        return errors

    def channel(self):
        """
        Generate the Kraus operators for a qubit system.
        """
        gamma, gamma1 = self._compute_transition_rates()
        kraus1, kraus2 = self._create_kraus_operators_qubit(gamma, gamma1)
        errors1 = self._create_error_channels(kraus1)
        errors2 = self._create_error_channels(kraus2)
        return errors1, errors2

    def channel_qutrit(self):
        """
        Generate the Kraus operators for a qutrit system.
        """
        gamma, gamma1 = self._compute_transition_rates()
        kraus_ops = self._create_kraus_operators_qutrit(gamma, gamma1)
        errors = self._create_error_channels(kraus_ops)
        return errors, errors

    def channel_ququart(self):
        """
        Generate the Kraus operators for a ququart system.
        """
        gamma, gamma1 = self._compute_transition_rates()
        kraus_ops = self._create_kraus_operators_ququart(gamma, gamma1)
        errors = self._create_error_channels(kraus_ops)
        return errors, errors
