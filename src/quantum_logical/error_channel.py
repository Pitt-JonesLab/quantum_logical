# Ampltiude Damping Channel

# TODO implemented Generalized Amplitude Damping Channel
# this considers "hot"-qubits, ground state is not perfectly |0>

from abc import ABC

import numpy as np
from qutip import Qobj


class ErrorChannel(ABC):
    def __init__(self, trotter_step_size):
        self.t = trotter_step_size

    def apply(self, state):
        return sum([E * state * E.dag() for E in self.E])


class AmplitudeDamping(ErrorChannel):
    def __init__(self, T1, trotter_step_size):
        super().__init__(trotter_step_size)
        _gamma = 1 - np.exp(-self.t / T1)
        E0 = Qobj([[1, 0], [0, np.sqrt(1 - _gamma)]])
        E1 = Qobj([[0, np.sqrt(_gamma)], [0, 0]])
        self.E = [E0, E1]


class PhaseDamping(ErrorChannel):
    def __init__(self, T2, trotter_step_size):
        super().__init__(trotter_step_size)
        _lambda = 1 - np.exp(-self.t / T2)
        E0 = Qobj([[1, 0], [0, np.sqrt(1 - _lambda)]])
        E1 = Qobj([[0, 0], [0, np.sqrt(_lambda)]])
        self.E = [E0, E1]
