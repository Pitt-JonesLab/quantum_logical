"""Composite interactions for the quantum logical model."""
# TODO
# composed interactions for VGate, DeltaGate

# class ComposedInteraction:
#     def __init__(self, interactions: List[ConversionGainInteraction], num_qubits: int):
#         self.num_qubits = num_qubits
#         self.interactions = interactions

#     def construct_H(self) -> Hamiltonian:
#         H = 0
#         for interaction in self.interactions:
#             H += interaction.construct_H(self.num_qubits).H
#         return Hamiltonian(H)


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
