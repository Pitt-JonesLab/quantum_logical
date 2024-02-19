"""building the hamiltonian for the system"""

import abc as ABC
import numpy as np
from quantum_logical.scqubits.modes import Modes
import scqubits as scq
import qutip as qt

class Hamiltonian(Modes):
    def __init__(self, modes):
        #takes in quantum modes and builds the hamiltonian
        self.modes = modes

    def hilbertspace_build(self):
        scq.HilbertSpace(self.modes)
        return 0
    
    def static(self):
        order = [i for i in self.modes]
        self.H = 0
        for i in self.modes:
            if(isinstance(i,Modes)):
                order[i] = self.Ej * scq.Transmon.n_operator
            if(isinstance == 'cavity' or self.type == 'oscillator'):
                order[i] = self.E_osc * scq.Oscillator.n_operator
        for w in order:
            self.H += w 

                
