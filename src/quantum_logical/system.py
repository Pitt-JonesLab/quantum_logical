"""Class representing a quantum system."""

from functools import reduce
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import qutip as qt
import yaml

from quantum_logical.hamiltonian import Hamiltonian
from quantum_logical.mode import QuantumMode
from functools import lru_cache


class QuantumSystem:
    """Class representing a quantum system.

    Rresponsible for managing the composite Hilbert space, including the
    creation and management of tensor products for mode operators and
    states.
    """

    def __init__(
        self,
        modes: List[QuantumMode],
        couplings: Dict[Tuple[QuantumMode], float],
        hamiltonian_cls: Hamiltonian = None,
    ):
        """Initialize a quantum system with given modes and couplings.

        Args:
            modes (list): A list of QuantumMode instances.
            couplings (dict): A dictionary where keys are tuples of coupled QuantumMode instances,
                              and values are the coupling strengths.
            hamiltonian_cls (Hamiltonian, optional): The Hamiltonian object to be used for the system.
        """
        self.modes = modes
        self.couplings = couplings
        (
            self.modes_a,
            self.modes_a_dag,
            self.modes_num,
            self.modes_field,
            self.modes_Z,
        ) = (
            None,
            None,
            None,
            None,
            None,
        )
        self._initialize_mode_operators()
        if hamiltonian_cls is None:
            hamiltonian_cls = Hamiltonian
        self.hamiltonian = hamiltonian_cls(self)

    def _initialize_mode_operators(self):
        """Initialize transformed system ops for each mode in the system.

        Creates and stores annihilators, number operators, and field
        operators for each mode.
        """
        self.modes_a = {mode: self._tensor_op(mode, mode.a) for mode in self.modes}
        self.modes_a_dag = {
            mode: self._tensor_op(mode, mode.a_dag) for mode in self.modes
        }
        self.modes_num = {mode: self._tensor_op(mode, mode.num) for mode in self.modes}
        self.modes_field = {
            mode: self._tensor_op(mode, mode.field) for mode in self.modes
        }
        self.modes_Z = {mode: self._tensor_op(mode, mode.Z) for mode in self.modes}

    def _tensor_op(self, mode: QuantumMode, op: qt.Qobj):
        """Tensor an operator with the identity operator on all other modes.

        Args:
            mode (QuantumMode): The mode for which the operator is defined.
            op (qutip.Qobj): The operator to be tensored.

        Returns:
            qutip.Qobj: The tensored operator in the Hilbert space of the entire system.
        """
        mode_index = self.modes.index(mode)
        op_list = [qt.qeye(m.dim) for m in self.modes]
        op_list[mode_index] = op
        return reduce(qt.tensor, op_list)

    def prepare_fock_state(self, mode_states: List[Tuple[QuantumMode, int]]):
        """Prepare a dressed Fock product state for specified modes.

        Args:
            mode_states (list of tuples): Each tuple contains a QuantumMode object and an integer
                                          representing the Fock state number for that mode.
                                          Modes not included in the list are assumed to be in the 0 state.

        Returns:
            qutip.Qobj: Tensor product state as a QuTiP object.
        """
        state_list = [qt.basis(mode.dim, 0) for mode in self.modes]
        for mode, state in mode_states:
            if mode not in self.modes:
                raise ValueError(f"Mode {mode} not found in system.")
            state_list[self.modes.index(mode)] = qt.basis(mode.dim, state)

        psi0 = reduce(qt.tensor, state_list)
        return psi0

    # def mode_population_expectation(
    #     self, system_state: qt.Qobj, mode: QuantumMode, fock_state: int
    # ):
    #     """Calculate expectation of the population of a mode in a given state.

    #     Args:
    #         system_state (qutip.Qobj): The state of the entire quantum system.
    #         mode (QuantumMode): The mode for which the population is calculated.
    #         fock_state (int): The Fock state number for the mode.

    #     Returns:
    #         float: The expectation value of the mode population.
    #     """
    #     fock_state = qt.basis(mode.dim, fock_state)
    #     fock_state_op = fock_state * fock_state.dag()
    #     system_fock_state_op = self._tensor_op(mode, fock_state_op)
    #     return qt.expect(system_fock_state_op, system_state)

    def __repr__(self) -> str:
        """Return a string representation of the QuantumSystem."""
        return f"QuantumSystem({self.modes})"

    @classmethod
    def from_yaml(cls, file_path: str):
        """Create a QuantumSystem instance from a YAML file.

        Args:
            file_path (str): Path to the YAML file containing modes and couplings data.

        Returns:
            QuantumSystem: An instance of QuantumSystem initialized from the YAML file.
        """
        yaml_path = Path(file_path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found at: {yaml_path}")

        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        modes = []
        for name, properties in data["modes"].items():
            mode = QuantumMode(name=name, **properties)
            modes.append(mode)

        # Process couplings
        couplings = {}
        for _, coupling in data["couplings"].items():
            mode_names = coupling["modes"]
            mode_objs = [
                next(mode for mode in modes if mode.name == name) for name in mode_names
            ]

            # Convert g2 from GHz to rad/s
            couplings[tuple(mode_objs)] = coupling["g2"] * 2 * np.pi

        return cls(modes, couplings)


class DressedQuantumSystem(QuantumSystem):
    """QuantumSystem with auxillary methods for hybridization of modes."""

    def _g(self, x, y):
        """Return the coupling strength between two modes."""
        return self.couplings.get(frozenset([x, y]), 0)

    def _lambda(self, x, y):
        """Return the hybridization strength between two modes."""
        # lambda = g / (omega_x - omega_y)
        if x is y:
            # FIXME, why negative?
            # because we use it like this (dressed_freq -= ...)
            return -1

        return -1 * self._g(x, y) / (x.freq - y.freq)

    def op_basis_transform(self, operator: qt.Qobj):
        """Transform an operator or density matrix to the dressed basis."""
        # NOTE, not entirely certain why this first method doesn't work
        # dressed_op = self.hamiltonian.H0 * operator * self.hamiltonian.H0.dag()
        # return dressed_op.unit()

        _, eigenvectors = self.hamiltonian.H0.eigenstates()

        dressed_rho = 0
        for eigenvector_i in eigenvectors:
            for eigenvector_j in eigenvectors:
                dressed_rho += (
                    (eigenvector_j.dag() * operator * eigenvector_i)
                    * eigenvector_i
                    * eigenvector_j.dag()
                )
        return dressed_rho

    @lru_cache(maxsize=None)
    def prepare_approx_state(self, mode_states: List[Tuple[QuantumMode, int]]):
        """Prepare a dressed state for the system.

        Rather than doing a strict basis transform, understand that
        physical states have an always-on hybridization between modes.
        This function prepares a state which is the mapping of the non-
        interacting state to the interacting state.

        Assumes that the always-on hybridization is weak enough that the
        state is still mostly in the non-interacting basis and therefore
        the inner product will be close to 1.
        """
        psi0 = super().prepare_fock_state(mode_states)
        # search for dressed initial state using the eigenstates of the Hamiltonian
        _, eigenvectors = self.hamiltonian.H0.eigenstates()

        best_fit = max(eigenvectors, key=lambda x: np.abs(x.overlap(psi0)))
        overlap = np.abs(best_fit.overlap(psi0))
        print(f"Found overlap with eigenstate by {overlap:.4f}")
        return best_fit

    def _initialize_dressed_freqs(self):
        r"""Calculate the dressed frequencies for each mode in the system.

        \Tilde{\omega}_x \approx \omega_x + \sum_y \lambda_{xy}g_{xy}

        Returns:
            dict: A dictionary with mode names as keys and dressed frequencies as values.
        """
        self.dressed_freqs = {}
        for mode in self.modes:
            dressed_freq = mode.freq
            for other_mode in self.modes:
                if mode != other_mode:
                    # += \lambda_{xy}g_{xy}
                    dressed_freq -= self._lambda(mode, other_mode) * self._g(
                        mode, other_mode
                    )
            self.dressed_freqs[mode] = dressed_freq
        return self.dressed_freqs
