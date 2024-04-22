import qutip as qt
from quantum_logical.pulsesim import QuantumSystem
from quantum_logical.pulsesim.mode import QubitMode, SNAILMode
import matplotlib.pyplot as plt
from itertools import product
from tqdm.notebook import tqdm




class Build_hamiltonian():
    # the first and second are the ones that are involved in driving
    # pass in the operators 
    def __init__(self, l1, qs, first, second, third, fourth):

        self.first = first
        self.qs = qs
        self.l1 = l1
        self.second = second
        self.third = third
        self.fourth = fourth

    def build_drive_hamiltonian(self):
        H_no_time = 6*(self.l1**2)*(self.qs.modes_a[self.first]*self.qs.modes_a_dag[self.second] + self.qs.modes_a[self.second]*self.qs.modes_a_dag[self.first])

        #terms that come from the gate that is desired that do not go to one
        qubit1_qubit2_adj_H = 6*(self.l1**2)*self.qs.modes_a[self.first]*self.qs.modes_a_dag[self.second]
        qubit1_adj_qubit2_H = 6*(self.l1**2)*self.qs.modes_a[self.second]*self.qs.modes_a_dag[self.first]

        H_main_qubits = [
        H_no_time,
        qubit1_qubit2_adj_H,
        qubit1_adj_qubit2_H
        ]

        return H_main_qubits
    
    def build_added(self):
        qubit3_qubit2_adj_H = self.qs.modes_a[self.third]*self.qs.modes_a_dag[self.second]
        qubit3_adj_qubit2_H = self.qs.modes_a[self.second]*self.qs.modes_a_dag[self.third]
        qubit4_qubit2_adj_H = self.qs.modes_a[self.fourth]*self.qs.modes_a_dag[self.second]
        qubit4_adj_qubit2_H = self.qs.modes_a[self.second]*self.qs.modes_a_dag[self.fourth]
        qubit3_qubit1_adj_H = self.qs.modes_a[self.third]*self.qs.modes_a_dag[self.first]
        qubit3_adj_qubit1_H = self.qs.modes_a[self.first]*self.qs.modes_a_dag[self.third]
        qubit4_qubit1_adj_H = self.qs.modes_a[self.fourth]*self.qs.modes_a_dag[self.first]
        qubit4_adj_qubit1_H = self.qs.modes_a[self.first]*self.qs.modes_a_dag[self.fourth]
        qubit4_adj_qubit3_H = self.qs.modes_a[self.third]*self.qs.modes_a_dag[self.fourth]
        qubit4_qubit3_adj_H = self.qs.modes_a[self.fourth]*self.qs.modes_a_dag[self.third]

        H_added = [
        qubit3_qubit2_adj_H,
        qubit3_adj_qubit2_H,
        qubit4_qubit2_adj_H,
        qubit4_adj_qubit2_H,
        qubit3_qubit1_adj_H,
        qubit3_adj_qubit1_H,
        qubit4_qubit1_adj_H,
        qubit4_adj_qubit1_H,
        qubit4_adj_qubit3_H,
        qubit4_qubit3_adj_H
        ]


        return H_added
    
