from qutip import Qobj
from qutip import tensor, qeye
import numpy as np
from itertools import product


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
        return Qobj(gate, dims=[dims, dims])
    
class Convert_levels():
        "Converting levels of the gate i.e. e -> g or vice versa"
        "there has to be something done for qubit specific gates instead of ones that just convert it for all of the qubits"
        def __init__(self, num_qubits):
            self.num_qubits = num_qubits
            
            return None
        
        def level_conversion(self, levels, dim, gate, qubits):
            # if qubits == None:
            single_qubit_matrix = np.zeros((dim, dim))
            for i in range(dim):  # populates the matrix with ones
                if i not in levels:
                    single_qubit_matrix[i, i] = 1
            
            single_qubit_matrix[levels[0], levels[1]] = single_qubit_matrix[levels[1], levels[0]] = 1
            switch_matrix = tensor([Qobj(single_qubit_matrix)] * self.num_qubits)

            gate = switch_matrix.dag() * gate * switch_matrix

            # this part below is a bit useless 
            # else:
            #     identity_op = [qeye(dim) for _ in range(self.num_qubits)]
            #     for qubit in qubits:
            #         single_qubit_matrix = np.zeros((dim, dim))
            #         for i in range(dim):  # populates the matrix with ones
            #             if i not in levels:
            #                 single_qubit_matrix[i, i] = 1
                    
            #         single_qubit_matrix[levels[0], levels[1]] = single_qubit_matrix[levels[1], levels[0]] = 1
            #         identity_op[qubit] = Qobj(single_qubit_matrix)
            #         switch_matrix = tensor(identity_op)

            #     gate = switch_matrix.dag() * gate * switch_matrix
            
            # create a vector system that abides by the qubits
            



            return gate
        
        def Cnot(self, dim, target, control, high, low):
                    
            # setting the control 
            control_vec = Qobj(np.array([1 if i == high else 0 for i in range(dim)]))
            # setting the target 
            targ_high_vec = Qobj(np.array([1 if i == high else 0 for i in range(dim)]))
            targ_low_vec = Qobj(np.array([1 if i == low else 0 for i in range(dim)]))
            set_targ_vecs = [targ_low_vec, targ_high_vec]
            # setting the spectators 
            spec_vec = []
            for i in range(dim):
                new_vec =  Qobj(np.array([0 if value != i else 1 for value in range(dim)]))
                spec_vec.append(new_vec)

            # the issue that i am having has to do with the spectator vectors function 

            spectator_vectors = list(product(spec_vec, repeat=(self.num_qubits-2)))
            # creating the matrices based on these vectors 
            set_of_vectors = []
            # for _ in range(len(spectator_vectors)):
            for w in range(len(spectator_vectors)):
                for targ_vec in set_targ_vecs:
                    qubit_vecs = [Qobj(np.array([0 for _ in range(dim)])) for _ in range(self.num_qubits)]
                    qubit_vecs[control] = control_vec
                    qubit_vecs[target] = targ_vec
                    k = 0
                    for i in range(self.num_qubits):
                        if i not in [control, target]:
                            # need to iterate over the place that zero is holding right now 
                            qubit_vecs[i] = spectator_vectors[w][k] # this does not make sense 
                            k +=1
                    set_of_vectors.append(tensor(qubit_vecs))

            # trace this out and see if it does what you want it to 

            # filter out the set of vectors to make sure that there are not any that are the same
            new_vectors = []
            new_vectors.append(set_of_vectors[0])
            for i in range(len(set_of_vectors)):
                if all(set_of_vectors[i] != new_vectors[j] for j in range(len(new_vectors))):
                    new_vectors.append(set_of_vectors[i])


            # first combine the even and odd entries separately
            set_of_matrices = []
            for i in range(0, len(new_vectors), 2):
                matrix = new_vectors[i] * new_vectors[i+1].dag()
                matrix = matrix + matrix.dag()
                set_of_matrices.append(matrix)

            full_set = np.zeros((dim ** self.num_qubits, dim ** self.num_qubits))
            for matrix in set_of_matrices:
                full_set = full_set + matrix.full()

            full_mat = np.zeros((dim ** self.num_qubits, dim ** self.num_qubits))
            i = 0
            for row in full_set:
                if any(np.abs(value) != 0 for value in row):
                    i += 1
                else:
                    full_mat[i,i] = 1
                    i += 1
                     
            gate = full_mat

            for matrices in set_of_matrices:
                gate = gate + matrices.full()

            gate = Qobj(gate, dims=([dim] * self.num_qubits, [dim] * self.num_qubits))

            return gate