import qutip as qt
from qutip import tensor, basis
import numpy as np
from quantum_logical.trotterization import Trotterization
# there is circular importation that needs fixed 


class Kraus_operators(Trotterization):
    def __init__(self, trotter_dt, T1, T2, dim, num_qubits, qudits):
        super().__init__(trotter_dt, T1, T2, dim, num_qubits, qudits)
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


        kraus1 = [a, b]
        kraus2 = [c, d]
        from itertools import zip_longest
        # creating identity built on the size of qubits but not yet tensored together
        for i in range(self.num_qubits):
            for k, j in zip_longest(kraus1, kraus2):
                identity = [qt.qeye(self.dim) for _ in range(self.num_qubits)]
                identity2 = [qt.qeye(self.dim) for _ in range(self.num_qubits)]
                identity[i] = k
                identity2[i] = j
                Errors.append(1 / np.sqrt(self.num_qubits) * tensor(identity))
                Errors1.append(1 / np.sqrt(self.num_qubits) * tensor(identity2))

        

        return Errors, Errors1
    
    def channel_qutrit(self):
        "build the same kraus operators but this time for a qutrit"

        # Simplified transition rates
        _fe = _eg = 1 - np.exp(-self.trotter_dt / self.T1)
        _fg = 0  # Neglected direct transition from f to g

        # Simplified Kraus operators for amplitude dampening only 
        A_01 = np.sqrt(_eg) * np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        A_12 = np.sqrt(_fe) * np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        A_02 = np.sqrt(_fg) * np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
        A_0 = (
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
            + np.sqrt(1 - _eg) * np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            + np.sqrt(1 - _fg - _fe)
            * np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        )
        Errors = []
        kraus_ops = [qt.Qobj(A_0), qt.Qobj(A_01), qt.Qobj(A_12), qt.Qobj(A_02)]
        for i in range(self.num_qubits):
            for j in kraus_ops:
                identity = [qt.qeye(self.dim) for _ in range(self.num_qubits)]
                identity[i] = j
                Errors.append(1 / np.sqrt(self.num_qubits) * tensor(identity))

        E_0 = qt.Qobj([[1,0,0],[0, np.sqrt(1-_fe), 0],[0,0,np.sqrt(1-_fe)]])
        E_1 = qt.Qobj([[0,0,0],[0,np.sqrt(_fe), 0],[0,0,0]])
        E_2 = qt.Qobj([[0,0,0],[0,0, 0],[0,0,np.sqrt(_fe)]])
        Errors2 = []
        Kraus_ops2 = [E_0, E_1, E_2]
        for i in range(self.num_qubits):
            for j in Kraus_ops2:
                identity = [qt.qeye(self.dim) for _ in range(self.num_qubits)]
                identity[i] = j
                Errors2.append(1 / np.sqrt(self.num_qubits) * tensor(identity))
        
        return Errors, Errors2

    def channel_ququart(self):
        # Simplified transition rates
        _hf = _fe = _eg = 1 - np.exp(-self.trotter_dt / self.T1)
        _hg = _he = _fg = 0  # Neglected direct transition from f to g
        # need to have one level transitions
        # Simplified Kraus operators for amplitude dampening only 
        A_01 = np.sqrt(_eg) * np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        A_12 = np.sqrt(_fe) * np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        A_02 = np.sqrt(_fg) * np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        A_03 = np.sqrt(_hg) * np.array([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        A_13 = np.sqrt(_he) * np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        A_23 = np.sqrt(_hf) * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        A_0 = (
            np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            + np.sqrt(1 - _eg) * np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
            + np.sqrt(1 - _fg - _fe)
            * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
            + np.sqrt(1 - _hg - _he - _hf)
            * np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
        )
        Errors = []
        kraus_ops = [qt.Qobj(A_0), qt.Qobj(A_01), qt.Qobj(A_12), qt.Qobj(A_02), qt.Qobj(A_03), qt.Qobj(A_13), qt.Qobj(A_23)]
        for i in range(self.num_qubits):
            for j in kraus_ops:
                identity = [qt.qeye(self.dim) for _ in range(self.num_qubits)]
                identity[i] = j
                Errors.append(1 / np.sqrt(self.num_qubits) * tensor(identity))

        E_0 = qt.Qobj([[1, 0, 0, 0],[0, np.sqrt(1-_hf), 0, 0],[0, 0, np.sqrt(1-_hf), 0], [0, 0, 0, np.sqrt(1-_hf)]])
        E_1 = qt.Qobj([[0, 0, 0, 0],[0, np.sqrt(_hf), 0, 0],[0, 0, 0, 0], [0, 0, 0, 0]])
        E_2 = qt.Qobj([[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, np.sqrt(_hf), 0], [0, 0, 0, 0]])
        E_3 = qt.Qobj([[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0], [0, 0, 0, np.sqrt(_hf)]])

        Errors2 = []
        Kraus_ops2 = [E_0, E_1, E_2, E_3]
        for i in range(self.num_qubits):
            for j in Kraus_ops2:
                identity = [qt.qeye(self.dim) for _ in range(self.num_qubits)]
                identity[i] = j
                Errors2.append(1 / np.sqrt(self.num_qubits) * tensor(identity))
        
        return Errors, Errors2