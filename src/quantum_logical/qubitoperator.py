from enum import Enum

import numpy as np
import qutip


class Transition(Enum):
    GE = "g->e"
    EF = "e->f"
    GF = "g->f"
    EG = "e->g"
    FE = "f->e"
    FG = "f->g"

    def dag(self):
        _dag_map = {
            "GE": "EG",
            "EF": "FE",
            "GF": "FG",
            "EG": "GE",
            "FE": "EF",
            "FG": "GF",
        }

        return Transition[_dag_map[self.name]]


class QubitOperator:
    def __init__(
        self, qubit_label: str, transition_type: Transition, transmon_levels: int = 2
    ):
        self.qubit_label = qubit_label
        self.transition_type = transition_type
        self.transmon_levels = transmon_levels

    @classmethod
    def from_qobj(cls, qobj: qutip.Qobj, qubit_label: str, transition_type: Transition):
        return cls(qubit_label, transition_type, transmon_levels=qobj.shape[0])

    def dag(self):
        return QubitOperator(
            self.qubit_label, self.transition_type.dag(), self.transmon_levels
        )

    def to_qutip(self) -> qutip.Qobj:
        op_matrix = np.zeros(
            (self.transmon_levels, self.transmon_levels), dtype=complex
        )

        if self.transition_type in {Transition.GE, Transition.EG}:
            op_matrix[1, 0] = 1
        elif self.transition_type in {Transition.EF, Transition.FE}:
            if self.transmon_levels > 2:
                op_matrix[2, 1] = 1
        elif self.transition_type in {Transition.GF, Transition.FG}:
            if self.transmon_levels > 2:
                op_matrix[2, 0] = 1

        op = qutip.Qobj(op_matrix)
        return (
            op.dag()
            if self.transition_type in {Transition.EG, Transition.FE, Transition.FG}
            else op
        )

    def __str__(self) -> str:
        return
