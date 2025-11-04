"""Trotterization of continuous operators."""

from typing import Iterable

import numpy as np
from qutip import Qobj
from scipy.linalg import fractional_matrix_power

from quantum_logical.channel import Channel, CPTPMap

__all__ = ["TrotterGroup"]

class TrotterGroup:
    def __init__(self, continuous_operators: Iterable[CPTPMap], trotter_dt):
        """Initialize with list of continuous ops and a Trotter step size.

        Continuous operators are applied continuously and accumulate
        over time.
        """
        self.trotter_dt = trotter_dt
        self.continuous_operators = []
        for op in continuous_operators:
            self._compose(op)

    def _compose(self, operator):
        """Compose a new operator into the TrotterGroup."""
        # Dimensionality check
        if self.continuous_operators:
            if operator.dims != self.continuous_operators[0].dims:
                raise ValueError("Dimension mismatch among operators in TrotterGroup.")
        elif isinstance(operator, Channel):
            if operator._trotter_dt != self.trotter_dt:
                operator.set_trotter_dt(self.trotter_dt)
            self.continuous_operators.append(operator)
        elif isinstance(operator, Qobj) and operator.isunitary:
            self.continuous_operators.append(operator)
        else:
            raise ValueError("Invalid operator type.")
    def apply(self, state, duration, unitaries):
        # put state into array not qobj 
        state = state.full() if isinstance(state, Qobj) else state

        # make sure the unitaries make sense
        for i in range(len(unitaries)):
            if not Qobj(unitaries[i]).isunitary:
                raise ValueError("Discrete unitary must be unitary.")
            elif isinstance(unitaries[i], Qobj):
                unitaries[i] = unitaries[i].full()
        
        num_step = (duration, self.trotter_dt)

        # create the unitary fraction list
        unitary_fraction = []
        for i in range(len(unitaries)):
            unitary_fraction.append(fractional_matrix_power(unitaries[i], (1 / num_step)))
        
        # apply the unitaries
        states = []
        if duration == 0:
            for i in range(len(unitaries)):
                state = unitaries[i] @ state @ unitaries[i].conj().T
            
        else:
            for _ in range(num_step):
                for op in self.continuous_operators:
                    state_numpy = op(state_numpy)

                for i in range(len(unitary_fraction)):
                    state = unitary_fraction[i] @ state @ unitary_fraction[i].conj().T

                states.append(Qobj(state_numpy, dims=state.dims) / np.trace(state_numpy))
        return states




