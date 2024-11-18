from qutip import Qobj
from qutip import tensor
import numpy as np


class Gate_extender():
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits

        return None

    def isometry(self, from_dim, to_dim):
        matrix = np.zeros((from_dim, to_dim))
        for i in range(from_dim):
            matrix[i, i] = 1

        return Qobj(matrix)
    
    def qubit_to_qudit(self, gate, from_dim, to_dim):
        "extends the gate into a larger subspace"

        iso = self.isometry(from_dim=from_dim, to_dim=to_dim)

        converter = tensor([iso for _ in range(self.num_qubits)])
        gate = (converter.dag() * gate * converter).full()  # extends into larger space

        for i in range(len(gate)):  # fills in the remaining values
            val = (np.abs(gate[i]) != 0).any()
            if val == False:
                gate[i][i] = 1

        dims = [to_dim for _ in range(self.num_qubits)]
        return Qobj(gate, dims=[dims, dims]), to_dim
    
class Convert_levels():
        "Converting levels of the gate i.e. e -> g or vice versa"
        def __init__(self, num_qubits):
            self.num_qubits = num_qubits
            
            return None
        
        def level_conversion(self, levels, dim, gate):
            single_qubit_matrix = np.zeros((dim, dim))
            for i in range(dim):  # populates the matrix with ones
                if i not in levels:
                    single_qubit_matrix[i, i] = 1
            
            single_qubit_matrix[levels[0], levels[1]] = single_qubit_matrix[levels[1], levels[0]] = 1
            switch_matrix = tensor([Qobj(single_qubit_matrix)] * self.num_qubits)

            gate = switch_matrix.dag() * gate * switch_matrix

            return gate