"""Contains a simulator for quantum circuits.

The simulator is designed to work with our custom research project.
In particular:
0. The simulator and error channels are designed to handle **qudits**.
1. *Trotterization* is used to approximate continuous-time evolution.
2. Fractional unitaries are computed from pulses/Hamiltonians.
3. Handles mid-circuit measurement and reset.
4. Assumes fidelity is coherence-limited, thus error source is circuit duration.

The input is a QuantumCircuit object, which has been compiled to run on this simulator.
For our purposes, this means:

0. Transformed from a logical circuit to an encoded circuit.
    - Logical qubits are comprised of multiple physical qubits.
    - Syndrome extractions and measurements are included in the circuit.
1. The circuit is scheduled using instruction set (each gate has a duration).

Reference: https://github.com/Qiskit/qiskit/blob/main/qiskit/providers/basicaer/qasm_simulator.py
"""

from math import log

from qiskit.providers.basicaer import QasmSimulatorPy
from qiskit.utils.multiprocessing import local_hardware_info


class QuditTrotterSim(QasmSimulatorPy):
    """Trotter-sim subclasses Qiskit QasmSimulatorPy."""

    pass


def max_qudits(d):
    """Calculate the max d-level qudits that can be stored in memory.

    :param d: Dimension of the quantum system (2 for qubits, 3 for
        qutrits, etc.).
    :return: Maximum number of qudits.
    """
    memory_in_bytes = local_hardware_info()["memory"] * (1024**3)

    # Assuming each state vector entry requires b bytes of storage.
    # This value depends on the implementation and may need to be adjusted.
    b = 16  # This is a placeholder and should be adjusted based on actual storage requirements

    # Calculate the number of qudits
    return int(log(memory_in_bytes / b) / log(d))


if __name__ == "__main__":
    # Example usage for qutrits (d=3)
    max_qudits(3)
