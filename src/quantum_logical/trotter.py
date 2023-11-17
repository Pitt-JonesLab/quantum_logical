"""Trotterization of continuous operators."""
from typing import Iterable

import numpy as np
from qutip import Qobj
from scipy.linalg import fractional_matrix_power

from quantum_logical.channel import Channel, CPTPMap

__all__ = ["TrotterGroup"]


class TrotterGroup:
    """Group of operators for Trotterized application."""

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

        if isinstance(operator, TrotterGroup):
            self.continuous_operators.extend(operator.operators)
        elif isinstance(operator, Channel):
            if operator._trotter_dt != self.trotter_dt:
                operator.set_trotter_dt(self.trotter_dt)
            self.continuous_operators.append(operator)
        elif isinstance(operator, Qobj) and operator.isunitary:
            self.continuous_operators.append(operator)
        else:
            raise ValueError("Invalid operator type.")

    def apply(self, state, duration, discrete_unitary=None):
        """Apply the group of operators to the state over a specified duration.

        If a discrete unitary is provided, it will be divided into fractional
        parts based on the number of Trotter steps.

        Args:
            state (Qobj): State to be evolved, as a density matrix.
            duration (float): Duration of evolution.
            discrete_unitary (Qobj): Optional discrete unitary to apply.
        """
        # convert into numpy ndarray
        state_numpy = state.full() if isinstance(state, Qobj) else state

        if discrete_unitary is not None and not Qobj(discrete_unitary).isunitary:
            raise ValueError("Discrete unitary must be unitary.")
        if discrete_unitary is not None and isinstance(discrete_unitary, Qobj):
            discrete_unitary = discrete_unitary.full()

        # Dimensionality check for state
        if state.shape != self.continuous_operators[0]._E[0].shape:
            raise ValueError(
                "State dimensions do not match the operators in TrotterGroup."
            )

        num_steps = int(duration / self.trotter_dt)
        if num_steps == 0:
            if discrete_unitary is not None:
                return Qobj(discrete_unitary @ state_numpy @ discrete_unitary.T.conj())
            return state

        fractional_unitary = None
        if discrete_unitary is not None:
            fractional_unitary = fractional_matrix_power(
                discrete_unitary, 1 / num_steps
            )

        for _ in range(num_steps):
            # Apply each continuous operator
            for op in self.continuous_operators:
                state_numpy = op(state_numpy)

            # Apply fractional part of the discrete unitary
            if fractional_unitary is not None:
                state_numpy = (
                    fractional_unitary @ state_numpy @ fractional_unitary.T.conj()
                )

        return Qobj(state_numpy, dims=state.dims) / np.trace(state_numpy)
