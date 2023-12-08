"""Selectively destroy a particle in a quantum system."""

from itertools import combinations_with_replacement, product

import numpy as np
import qutip
import qutip as qt
from qutip import Qobj, basis, tensor
from weylchamber import c1c2c3

# from qiskit.circuit.library import XGate, CZGate

__all__ = [
    "transform_ge_to_gf_gate",
    "define_observables",
    "reduce_to_two_qubit_subspace",
    "selective_destroy",
    # "_qutrit_to_3coords",
]


def transform_ge_to_gf_gate(gate):
    """Transform a 1Q or 2Q qubit gate into a corresponding qutrit gate.

    In the qutrit system, the additional state |e> is treated as an inactive
    state.

    The transformation is defined as follows:
    - For a 1Q gate, we use the isometry U = |g><0| + |f><1|, applying it to the gate.
    - For a 2Q gate, we use the tensor product of the isometries and add the identity
      operations for states involving |e>.
    - The resulting gate G' is unitary and acts as an identity operation on the |e> state.

    Parameters:
    - gate: A qubit gate represented as a qutip Qobj (2x2 for 1Q, 4x4 for 2Q).

    Returns:
    - A qutip Qobj representing the transformed qutrit gate.
    """
    gate = qt.Qobj(gate)

    # Basis states for qutrits
    g, e, f = qt.basis(3, 0), qt.basis(3, 1), qt.basis(3, 2)

    # Define the isometry for a single qutrit
    U_single = qt.Qobj(np.array([[1, 0], [0, 0], [0, 1]]))

    if gate.shape == (2, 2):
        # For a 1Q gate
        transformed_gate = U_single * gate * U_single.dag() + e * e.dag()
    elif gate.shape == (4, 4):
        # For a 2Q gate
        U_double = qt.tensor(U_single, U_single)
        gate.dims = [[2, 2], [2, 2]]
        transformed_gate = U_double * gate * U_double.dag()
        # Correctly adding identity operations for states involving |e>
        for state in [g, f]:
            transformed_gate += qt.tensor(e * e.dag(), state * state.dag()) + qt.tensor(
                state * state.dag(), e * e.dag()
            )
        transformed_gate += qt.tensor(e * e.dag(), e * e.dag())
    else:
        raise ValueError("Input gate must be a 2x2 or 4x4 matrix.")

    return transformed_gate


# # Example usage
# x = XGate().to_matrix()
# cz = CZGate().to_matrix()
# X_transformed = transform_ge_to_gf_gate(x)
# CNOT_transformed = transform_ge_to_gf_gate(cz)

# print("Transformed X gate:\n", X_transformed)
# print("\nTransformed CNOT gate:\n", CNOT_transformed)


def define_observables(N, d=3, exclude_symmetric=True):
    """Generate observables for an N-qubit system.

    Args:
        N (int): The number of qubits in the system.
        d (int): The dimension of the Hilbert space for each qubit (default is 3 for qutrits).
        exclude_symmetric (bool): If True, generate unique combinations without regard to order.
                                  If False, generate all possible permutations considering the order.

    Returns:
        dict: A dictionary of observables keyed by their state labels.
        list: A list of observable state labels.
    """
    basis_states = [basis(d, i) for i in range(d)]
    state_labels = ["g", "e", "f"]  # Labels for the states

    observable_labels = []
    observables = {}

    # Generate combinations or permutations based on exclude_symmetric
    state_sequences = (
        combinations_with_replacement(state_labels, N)
        if exclude_symmetric
        else product(state_labels, repeat=N)
    )

    for combo in state_sequences:
        label = "".join(combo)
        observable_labels.append(label)
        state = tensor([basis_states[state_labels.index(i)] for i in combo])
        observables[label] = state * state.dag()

    return observables, observable_labels


def _qutrit_to_3coords(qutrit_unitary):
    """Convert a unitary operator in a three-level system to three coordinates.

    NOTE: it is unclear to me if this is in general physically meaningful.
    However, can be used to verify if two are identities, then the gate can be
    considered as a qubit gate between the non-identity levels.

    Parameters:
        qutrit_unitary (Qobj): A unitary operator in a three-level system.

    Returns:
        tuple: The three coordinates corresponding to the unitary.
    """
    u_ge = reduce_to_two_qubit_subspace(qutrit_unitary, [0, 1])
    u_ef = reduce_to_two_qubit_subspace(qutrit_unitary, [1, 2])
    u_gf = reduce_to_two_qubit_subspace(qutrit_unitary, [0, 2])

    # NOTE, c1c2c3() can still calculate coordinates for non-unitary matrices
    # but unclear to me how meaningful the result will be. I would expected matrices
    # that are nearly unitary but, for example, are not normalized to be
    # trace-preserving to have coordinates as if it was properly normalized.

    if not u_ge.isunitary:
        raise Warning("u_ge is not unitary")
    if not u_ef.isunitary:
        raise Warning("u_ef is not unitary")
    if not u_gf.isunitary:
        raise Warning("u_gf is not unitary")

    return c1c2c3(u_ge), c1c2c3(u_ef), c1c2c3(u_gf)


def reduce_to_two_qubit_subspace(unitary, indices):
    """Reduces a two-qudit U to a two-qubit U using isometric projection.

    Parameters:
        unitary (Qobj): A unitary operator in a two-qudit Hilbert space.
        indices (list): List of integer indices [index1, index2] that specify which
                        qudit levels to map to the |0> and |1> qubit states.

    Returns:
        Qobj: The unitary operator reduced to the two-qubit subspace.
    """
    if not isinstance(unitary, Qobj):
        raise ValueError("The unitary parameter must be a QuTiP Qobj.")

    if len(indices) != 2:
        raise ValueError(
            "Indices must be of length 2, specifying the levels to keep for each qudit."
        )

    # Infer the level dimension from the shape of the unitary
    level_dim = int(np.sqrt(unitary.shape[0]))
    if level_dim**2 != unitary.shape[0]:
        raise ValueError(
            "The unitary must be a square matrix corresponding to a two-qudit system."
        )

    # Validate indices
    if any(idx >= level_dim for idx in indices):
        raise ValueError("Index out of range for the given level dimension.")

    # Create the isometry
    isometry = Qobj(
        np.zeros((4, level_dim**2)), dims=[[2, 2], [level_dim, level_dim]]
    )

    # Construct the isometry using tensor products of the basis states
    for qubit_index, (qudit_index1, qudit_index2) in enumerate(
        product(indices, repeat=2)
    ):
        qudit_state = tensor(
            basis(level_dim, qudit_index1), basis(level_dim, qudit_index2)
        )
        isometry += (
            tensor(basis(2, qubit_index // 2), basis(2, qubit_index % 2))
            * qudit_state.dag()
        )

    # Apply the isometric projection to the unitary
    reduced_unitary = isometry * unitary * isometry.dag()

    # Ensure that the returned Qobj has the correct dimensions for a two-qubit system
    reduced_unitary.dims = [[2, 2], [2, 2]]

    return reduced_unitary


def selective_destroy(levels: int, from_level: int, to_level: int) -> qutip.Qobj:
    """Create a selective destruction operator for a specified transition.

    Parameters:
        levels (int): The total number of levels in the system.
        from_level (int): The level from which the particle is annihilated.
        to_level (int): The level to which the particle transitions after annihilation.

    Returns:
        Qobj: The selective annihilation operator as a QuTiP quantum object.

    Raises:
        ValueError: If the specified levels are out of range or the transition is invalid.
    """
    if not (0 <= from_level < levels) or not (0 <= to_level < levels):
        raise ValueError("Transition levels must be within the range of system levels.")
    if from_level <= to_level:
        raise ValueError(
            "Invalid transition. from_level must be greater than to_level for annihilation."
        )

    # Create the matrix for the annihilation operator
    op_matrix = np.zeros((levels, levels), dtype=complex)

    # XXX: In a harmonic oscillator, the matrix elements of the annihilation operator
    # are proportional to the square root of the level number (sqrt(n)).
    # For a transmon qubit or similar quantum systems where the transition probabilities
    # do not scale with sqrt(n), we use unit matrix elements to represent the transitions.
    # This is a simplification that may need to be revisited based on the specifics
    # of the physical system and the transitions being modeled.
    op_matrix[to_level, from_level] = 1

    return qutip.Qobj(op_matrix)


if __name__ == "__main__":
    # Example usage:
    transmon_levels = 4  # Number of levels
    from_level = 2  # Transition from level |2>
    to_level = 1  # Transition to level |1>

    # Create the selective destruction operator for the 2->1 transition
    a_selective = selective_destroy(transmon_levels, from_level, to_level)
    print(a_selective)
