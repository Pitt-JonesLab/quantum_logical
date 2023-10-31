from abc import ABC
from functools import reduce
from typing import List

import numpy as np
import qutip

from quantum_logical.qubitoperator import QubitOperator, Transition


class Hamiltonian:
    def __init__(self, H: qutip.Qobj):
        self.H = H

    def construct_U(self, t: float) -> qutip.Qobj:
        """Construct the unitary evolution operator for time t.

        Args:
            t (float): Time.

        Returns:
            qutip.Qobj: Unitary operator for the given time.
        """
        return (-1j * self.H * t).expm()


class ConversionGainInteraction(ABC):
    def __init__(self, terms, coefficients, transmon_levels=2):
        """Initialize the n-wave interaction parameters.

        Args:
            terms (list of list of str): List of operator terms like [['a', 'b_dag'], ['a_dag', 'b']].
            coefficients (list of complex): List of coefficients for each term.
            transmon_levels (int, optional): Number of transmon levels. Defaults to 2.
        """
        self.terms = terms
        self.coefficients = coefficients
        self.transmon_levels = transmon_levels

        # TODO make alphabetical
        self.qubit_to_index = {
            op[0]: idx
            for idx, op in enumerate(
                set(op.qubit_label for term in terms for op in term)
            )
        }

    def unitary(self, t: float) -> qutip.Qobj:
        """Construct the unitary evolution operator for time t.

        Args:
            t (float): Time.

        Returns:
            qutip.Qobj: Unitary operator for the given time.
        """
        return self.construct_H().construct_U(t)

    def construct_single_operator(self, qubit_op, num_qubits):
        op = qubit_op.to_qutip()
        operators = [qutip.qeye(self.transmon_levels) for _ in range(num_qubits)]
        qubit_index = self.qubit_to_index[qubit_op.qubit_label]
        operators[qubit_index] = op
        return qutip.tensor(operators)

    def construct_H(self) -> Hamiltonian:
        H = 0
        for term, coeff in zip(self.terms, self.coefficients):
            term_operators = [self.construct_single_operator(op, 2) for op in term]
            # FIXME, order matters here
            term_product = reduce(
                lambda x, y: x @ y.full(),
                term_operators,
                np.eye(self.transmon_levels**2),
            )
            term_product = qutip.Qobj(term_product)
            H += coeff * term_product
            H += np.conj(coeff) * term_product.dag()
        return Hamiltonian(H)

    # def __str__(self):
    #     """String representation of the Hamiltonian in LaTeX form.

    #     Returns:
    #         str: LaTeX string.
    #     """
    #     coeff_to_terms = defaultdict(list)
    #     for term, coeff in zip(self.terms, self.coefficients):
    #         operators = " ".join(
    #             [op.qubit_label.replace("_dag", r"^\dagger") for op in term]
    #         )
    #         coeff_str = f"{coeff.real:.2f}"
    #         if np.iscomplex(coeff):
    #             coeff_str += f" + {coeff.imag:.2f}i"
    #         coeff_to_terms[coeff_str].append(operators)

    #     latex_str = r"\hat{H} = "
    #     for coeff, terms in coeff_to_terms.items():
    #         latex_str += f"({coeff}) ("
    #         for term in terms:
    #             latex_str += f"{term} + "
    #         latex_str = latex_str.rstrip(" + ")
    #         latex_str += ") + "

    #     latex_str = latex_str.rstrip(" + ") + r" + \text{h.c.}"
    #     return latex_str


# Example child classes
class ConversionGainThreeWave(ConversionGainInteraction):
    def __init__(self, gc, gg, phi_c=0.0, phi_g=0.0, transmon_levels=2):
        terms = [
            [
                QubitOperator("a", Transition.GE, transmon_levels=transmon_levels),
                QubitOperator("b", Transition.GE, transmon_levels=transmon_levels),
            ],
            [
                QubitOperator("a", Transition.GE, transmon_levels=transmon_levels),
                QubitOperator("b", Transition.EG, transmon_levels=transmon_levels),
            ],
        ]
        coefficients = [gc * np.exp(1j * phi_c), gg * np.exp(1j * phi_g)]
        super().__init__(terms, coefficients, transmon_levels)


class ConversionGainThreeWaveTogether(ConversionGainInteraction):
    def __init__(self, gc, gg, phi_c1=0.0, phi_c2=0.0, phi_g=0.0, transmon_levels=3):
        terms = [
            [
                QubitOperator("a", Transition.GE, transmon_levels=transmon_levels),
                QubitOperator("b", Transition.FE, transmon_levels=transmon_levels),
            ],
            [
                QubitOperator("a", Transition.EG, transmon_levels=transmon_levels),
                QubitOperator("b", Transition.EF, transmon_levels=transmon_levels),
            ],
            [
                QubitOperator("a", Transition.EF, transmon_levels=transmon_levels),
                QubitOperator("b", Transition.EG, transmon_levels=transmon_levels),
            ],
            [
                QubitOperator("a", Transition.EF, transmon_levels=transmon_levels),
                QubitOperator("b", Transition.EF, transmon_levels=transmon_levels),
            ],
        ]
        coefficients = [
            gc * np.exp(1j * phi_c1),
            gg * np.exp(1j * phi_g),
            gc * np.exp(1j * phi_c2),
            gg * np.exp(1j * phi_g),
        ]
        super().__init__(terms, coefficients, transmon_levels)


class ConversionGainFiveWave(ConversionGainInteraction):
    def __init__(self, gc, gg, phi_c=0.0, phi_g=0.0, transmon_levels=3):
        terms = [
            [
                QubitOperator("a", Transition.GF, transmon_levels=transmon_levels),
                QubitOperator(
                    "b", Transition.GF, transmon_levels=transmon_levels
                ).dag(),
            ],
            [
                QubitOperator("a", Transition.GF, transmon_levels=transmon_levels),
                QubitOperator("b", Transition.FG, transmon_levels=transmon_levels),
            ],
        ]
        coefficients = [gc * np.exp(1j * phi_c), gg * np.exp(1j * phi_g)]
        super().__init__(terms, coefficients, transmon_levels)


# TODO
# composed interactions for VGate, DeltaGate


class ComposedInteraction:
    def __init__(self, interactions: List[ConversionGainInteraction], num_qubits: int):
        self.num_qubits = num_qubits
        self.interactions = interactions

    def construct_H(self) -> Hamiltonian:
        H = 0
        for interaction in self.interactions:
            H += interaction.construct_H(self.num_qubits).H
        return Hamiltonian(H)


# class ComposedInteraction:
#     def __init__(self, interactions: List[ConversionGainInteraction], num_qubits: int):
#         """
#         Initialize the parameters for composed interactions.

#         Args:
#             interactions (List[ConversionGainInteraction]): List of 2-qubit interactions.
#             num_qubits (int): Total number of qubits in the system.
#         """
#         self.num_qubits = num_qubits
#         self.interactions = interactions
#         self.qubit_to_index = self.build_qubit_index_map()

#     def build_qubit_index_map(self) -> dict:
#         """Build a mapping from qubit identifier to index in the full system."""
#         qubit_ids = set()
#         for interaction in self.interactions:
#             qubit_ids.update(interaction.qubit_to_index.keys())
#         return {qid: idx for idx, qid in enumerate(sorted(qubit_ids))}

#     def construct_H(self) -> Hamiltonian:
#         """
#         Construct the system Hamiltonian.

#         Returns:
#             Hamiltonian: The Hamiltonian for the full system.
#         """
#         H = 0
#         for interaction in self.interactions:
#             H += self.map_interaction_to_system(interaction)
#         return Hamiltonian(H)

#     def map_interaction_to_system(self, interaction: ConversionGainInteraction) -> qutip.Qobj:
#         """
#         Map a 2-qubit interaction Hamiltonian to the full system.

#         Args:
#             interaction (ConversionGainInteraction): The 2-qubit interaction.

#         Returns:
#             qutip.Qobj: The Hamiltonian mapped to the full system.
#         """
#         H_interaction = interaction.construct_H().H  # Extract the qutip.Qobj Hamiltonian
#         # Initialize full system operators as identity matrices
#         full_system_ops = [qutip.qeye(self.num_qubits) for _ in range(self.num_qubits)]

#         for qubit_id, qubit_idx in interaction.qubit_to_index.items():
#             full_system_idx = self.qubit_to_index[qubit_id]
#             full_system_ops[full_system_idx] = H_interaction

#         return qutip.tensor(full_system_ops)
