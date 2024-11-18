import qutip as qt
from qutip import tensor, basis
import numpy as np
from quantum_logical.trotterization import Trotterization

class Kraus_operators(Trotterization):
    def __init__(self, trotter_dt, T1, T2, dim, num_qubits):
        super().__init__(trotter_dt, T1, T2, dim, num_qubits)

        return None
    
    def channel(self):
        # trotter_dt = .001
        # T1 = 1
        gamma = 1 - np.exp(-self.trotter_dt/self.T1)
        T_phi = (1/(1/self.T2 - 1/(2 * self.T1)))
        gamma1 = 1 - np.exp(-self.trotter_dt/T_phi)
        a = qt.Qobj([[1,0],[0,np.sqrt(1 - gamma)]])
        b = qt.Qobj([[0,np.sqrt(gamma)],[0,0]])
        c = qt.Qobj([[1,0],[0,np.sqrt(1 - gamma1)]])
        d = qt.Qobj([[0,0],[0,np.sqrt(gamma1)]])
        # can make this a touch cleaner but may not be worth it time wise
        # create all of the error channels for t1 first 
        Errors = []
        Errors1 = []
        num_qubits = self.num_qubits


        kraus1 = [a, b]
        kraus2 = [c, d]
        from itertools import zip_longest
        # creating identity built on the size of qubits but not yet tensored together 
        for i in range(num_qubits):
            for k, j in zip_longest(kraus1, kraus2):
                identity = [qt.qeye(self.dim) for _ in range(num_qubits)]
                identity2 = [qt.qeye(self.dim) for _ in range(num_qubits)]
                identity[i] = k
                identity2[i] = j
                Errors.append(1 / np.sqrt(num_qubits) * tensor(identity))
                Errors1.append(1 / np.sqrt(num_qubits) * tensor(identity2))

        

        return Errors, Errors1