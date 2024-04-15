"""Qudit operations."""

import warnings

import numpy as np
from qiskit.circuit import ControlledGate, Gate
from qiskit.circuit.library import CXGate, XGate
from qiskit.extensions import UnitaryGate
from qiskit.extensions.exceptions import ExtensionError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

from quantum_logical.operators import transform_ge_to_gf_gate


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

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None):
        """Return controlled version of gate.

        Args:
            num_ctrl_qubits (int): number of controls to add to gate (default=1)
            label (str): optional gate label
            ctrl_state (int or str or None): The control state in decimal or as a
                bit string (e.g. '1011'). If None, use 2**num_ctrl_qubits-1.

        Returns:
            UnitaryGate: controlled version of gate.

        Raises:
            QiskitError: Invalid ctrl_state.
            ExtensionError: Non-unitary controlled unitary.
        """
        warnings.warn(
            "Controlled QutritUnitary class is hardcoded to only be CX_gf gate."
        )
        cx_gf = transform_ge_to_gf_gate(CXGate().to_matrix())
        cx_gf = QutritUnitary(cx_gf, "cx_gf")
        return cx_gf

        # return QutritControlledGate(
        #     "c-unitary",
        #     num_qubits=self.num_qubits + num_ctrl_qubits,
        #     params=[mat],
        #     label=label,
        #     num_ctrl_qubits=num_ctrl_qubits,
        #     definition=iso,  # .definition,
        #     ctrl_state=ctrl_state,
        #     base_gate=self.copy(),
        # )


class QutritControlledGate(ControlledGate):
    """Controlled version of a qutrit unitary gate."""

    def __init__(self, name, **kwargs):
        """Create a controlled gate."""
        raise NotImplementedError


# TODO verify endian-ness :)
if __name__ == "__main__":
    x_gf = transform_ge_to_gf_gate(XGate().to_matrix())
    x_gf = QutritUnitary(x_gf, "x_gf")
    x_gf.control(1)
