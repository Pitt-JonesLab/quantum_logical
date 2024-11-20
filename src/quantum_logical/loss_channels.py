import qutip as qt
from qutip import tensor, basis
import numpy as np
from quantum_logical.trotterization import Trotterization


# confirming the operation using the kraus operators 
# single qubit operation
# make this more general to different losses for different qubits 
# this should have now fixed the trotterization now you need to integrate it into the system 
# this needs rewritten it is so sloppy

class Loss_channel(Trotterization):
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



        # there has to be a much faster way to combine these 
        # create all of the error channels for t1 first 
        Errors = []
        Errors1 = []
        num_qubits = self.num_qubits
        for i in range(num_qubits):
            if i not in [0, num_qubits - 1]:
                Errors.append(1 / np.sqrt(num_qubits) * tensor(tensor([qt.qeye(self.dim)] * i), a, tensor([qt.qeye(self.dim)] * (num_qubits - i - 1))))
                Errors.append(1 / np.sqrt(num_qubits) * tensor(tensor([qt.qeye(self.dim)] * i), b, tensor([qt.qeye(self.dim)] * (num_qubits - i - 1))))
                Errors1.append(1 / np.sqrt(num_qubits) * tensor(tensor([qt.qeye(self.dim)] * i), c, tensor([qt.qeye(self.dim)] * (num_qubits - i - 1))))
                Errors1.append(1 / np.sqrt(num_qubits) * tensor(tensor([qt.qeye(self.dim)] * i), d, tensor([qt.qeye(self.dim)] * (num_qubits - i - 1))))
            elif i == 0:
                Errors.append(1 / np.sqrt(num_qubits) * tensor(a, tensor([qt.qeye(self.dim)] * (num_qubits - 1))))
                Errors.append(1 / np.sqrt(num_qubits) * tensor(b, tensor([qt.qeye(self.dim)] * (num_qubits - 1))))
                Errors1.append(1 / np.sqrt(num_qubits) * tensor(c, tensor([qt.qeye(self.dim)] * (num_qubits - 1))))
                Errors1.append(1 / np.sqrt(num_qubits) * tensor(d, tensor([qt.qeye(self.dim)] * (num_qubits - 1))))
            else:
                Errors.append(1 / np.sqrt(num_qubits) * tensor(tensor([qt.qeye(self.dim)] * (num_qubits - 1)), a))
                Errors.append(1 / np.sqrt(num_qubits) * tensor(tensor([qt.qeye(self.dim)] * (num_qubits - 1)), b))
                Errors1.append(1 / np.sqrt(num_qubits) * tensor(tensor([qt.qeye(self.dim)] * (num_qubits - 1)), c))
                Errors1.append(1 / np.sqrt(num_qubits) * tensor(tensor([qt.qeye(self.dim)] * (num_qubits - 1)), d))

            # Errors = [1 / np.sqrt(num_qubits) * error for error in Errors]
            # Errors1 = [1 / np.sqrt(num_qubits) * error for error in Errors1]
56
        return Errors, Errors1


