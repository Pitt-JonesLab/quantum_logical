"""Selectively destroy a particle in a quantum system."""

import numpy as np
import qutip


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
