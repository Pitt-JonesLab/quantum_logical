"""Selectively destroy a particle in a quantum system."""

from itertools import product

import numpy as np
import qutip
from qutip import Qobj, basis, tensor
from weylchamber import c1c2c3


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
