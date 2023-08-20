"""Implicit Unitary Gate.

The Implicit Unitary Gate abstraction represents certain quantum operators with implicit terms.
It's useful in scenarios like Logical Encoding Manifold and Error Detection with Ancilla.

1. Logical Encoding Manifold: Some terms may be 'don't care' states, e.g.,
   - Given: \( |01\rangle \langle 00| + |10\rangle \langle 11| \)
   - Implicitly: \( |01\rangle \langle 01| + |10\rangle \langle 10| \)
   since the unmentioned states are 'don't cares.'

2. Error Detection with Ancilla: The absence of certain terms indicates they do not affect the ancilla, e.g.,
   - Given: \( |1\rangle \otimes |error\_state\rangle  \langle 0| \otimes \langle error\_state| \)
   - Implicitly: \( |0\rangle \otimes |no\_error\_state\rangle \sum_{\text{{no-error states}}} \)
   since these don't raise the ancilla.

These implicit terms provide a compact notation for quantum error correction and logical state encoding.
"""

import numpy as np
from qiskit.extensions import UnitaryGate


class ImplicitUnitaryGate(UnitaryGate):
    """A unitary gate from an implicit definition, useful in logical encoding
    and error detection.

    The implicit rule associates missing bras with corresponding kets,
    adds reverse terms, and fills the diagonal terms of the matrix.
    """

    def __init__(self, implicit_operator):
        unitary_operator = self.create_unitary_from_implicit_operator(implicit_operator)
        super().__init__(unitary_operator)

    @staticmethod
    def create_unitary_from_implicit_operator(implicit_operator):
        """Create unitary operator from an implicit definition.

        Args:
            implicit_operator (np.ndarray): Operator with implicit terms in bra-ket notation.

        Returns:
            np.ndarray: Unitary matrix with the complete operator, following the implicit rule.
        """

        operator = np.copy(implicit_operator)

        # Add reverse terms (e.g., if |a><b| exists, add |b><a|)
        for i in range(operator.shape[0]):
            for j in range(operator.shape[1]):
                if i != j and operator[i, j] != 0:
                    if operator[j, i] == 0:
                        operator[j, i] = operator[i, j]

        # Fill missing diagonal terms for unitarity
        for i in range(operator.shape[0]):
            missing_term = 1 - sum(a * np.conj(a) for a in operator[i, :])
            operator[i, i] += missing_term

        return operator


# Example usage
if __name__ == "__main__":
    # Example: |1> \otimes |00> <0| \otimes <00|
    # Implicit: |0> \otimes |00> <1| \otimes <00|
    implicit_op = np.zeros((8, 8))
    implicit_op[4, 0] = 1
    ImplicitUnitaryGate(implicit_op)
