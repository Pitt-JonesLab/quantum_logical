import numpy as np
from qutip import Qobj


class Creating_operators():
    def __init__(self, qubit_dim, qubit_count):

        self.qubit_dim = qubit_dim
        self.qubit_count = qubit_count
        self.hilbert_dim = qubit_dim ** qubit_count
        return None

    def create_ops_unconventional_annhilation(self, from_level, to_level):

        op_matrix = np.zeros((self.qubit_dim, self.qubit_dim))

        # for i in range(self.qubit_dim):  # will make the identity matrix
        #     if i != from_level:
        #         op_matrix[i, i] = 1
        
        op_matrix[to_level, from_level] = 1
        # operator has been created
        # make this matrix into a quantum object 
        op_matrix = Qobj(op_matrix)
        return op_matrix
    
    def creating_ops_full_dim(self, qubit1_from, qubit1_to, qubit2_from, qubit2_to):

        op_matrix = np.zeros((self.hilbert_dim, self.hilbert_dim))
        # qubit 1 is the one that is furthest left in the tensor product 
        op_matrix[qubit1_to * self.qubit_dim + qubit2_to, qubit1_from * self.qubit_dim + qubit2_from] = 1

        return Qobj(op_matrix)

    def total_operator(self, coeff_list, ops_list):

        # self.coeff_list = coeff_list
        # self.ops_list = ops_list
        
        if len(coeff_list) == len(ops_list):
            ops_coeff = [i * j for i, j in zip(coeff_list, ops_list)]
        
        return sum(ops_coeff)
        # gives the total operator acting on the system of qudits 

    def propagator(self, H, T):
        return (-1j * H * T).expm()
    
    def ideal_cnot_gate(self, from_level, to_level, control_level):
        matrix = np.zeros((self.hilbert_dim, self.hilbert_dim))

        for i in range(self.hilbert_dim):
            if i != self.qubit_dim * control_level + to_level and i != self.qubit_dim * control_level + from_level:
                matrix[i,i] = 1  # makes an identity matrix (where the states are not expected to change based on the application of this operator)

        matrix[self.qubit_dim * control_level + to_level, self.qubit_dim * control_level + from_level] = 1
        matrix[self.qubit_dim * control_level + from_level, self.qubit_dim * control_level + to_level] = 1

        return  Qobj(matrix)
    
    def Create_coeff(self, phase_list, coeff_list):
        return [i * np.exp(-1j * k) for i, k in zip(coeff_list, phase_list)]
    
    def not_gate(self, exchange_levels):
        if len(exchange_levels) < self.qubit_dim:
            matrix = np.zeros((self.qubit_dim, self.qubit_dim))
            level_one = exchange_levels[0]
            level_two = exchange_levels[1]

            for i in range(self.qubit_dim):  # will make the identity matrix
                if i != level_one and i != level_two:
                    matrix[i, i] = 1

            matrix[level_two, level_one] = matrix[level_one, level_two] = 1
            return Qobj(matrix)
        else:
            print(f"the levels given do not make sense in relation the qubit dimensionality")

        
