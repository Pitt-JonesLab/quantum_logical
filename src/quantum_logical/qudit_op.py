"""Qudit operations."""
import numpy as np
from qiskit.circuit import Gate
from qiskit.extensions import UnitaryGate
from qiskit.extensions.exceptions import ExtensionError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix


class QutritUnitary(UnitaryGate):
    """Unitary gate on a qutrit system.

    Qiskit's UnitaryGate class only supports qubits. This class is a
    copy of UnitaryGate with the qubit-specific code removed.
    """

    # TODO: generalize to qudits
    dim = 3

    def __init__(self, data, label=None):
        """Create a gate from a numeric unitary matrix.

        Args:
            data (matrix or Operator): unitary operator.
            label (str): unitary name for backend [Default: None].

        Raises:
            ExtensionError: if input data is not an N-qubit unitary operator.
        """
        if hasattr(data, "to_matrix"):
            # If input is Gate subclass or some other class object that has
            # a to_matrix method this will call that method.
            data = data.to_matrix()
        elif hasattr(data, "to_operator"):
            # If input is a BaseOperator subclass this attempts to convert
            # the object to an Operator so that we can extract the underlying
            # numpy matrix from `Operator.data`.
            data = data.to_operator().data
        # Convert to numpy array in case not already an array
        data = np.array(data, dtype=complex)
        # Check input is unitary
        if not is_unitary_matrix(data):
            raise ExtensionError("Input matrix is not unitary.")
        # Check input is N-qubit matrix
        input_dim, output_dim = data.shape
        num_qubits = int(np.log(input_dim) / np.log(self.dim))
        if input_dim != output_dim or self.dim**num_qubits != input_dim:
            raise ExtensionError("Input matrix is not an N-qubit operator.")

        # Store instruction params
        Gate.__init__(self, "unitary", num_qubits, [data], label=label)
