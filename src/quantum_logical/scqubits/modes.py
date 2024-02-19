"""file that will be used to create the individual modes"""
import numpy as np 
import abc as ABC
import scqubits as scq

class Modes:
    def __init__(self, **kwargs):
        #defining all of the useful variables
        self.type = kwargs["type"]

    def mode_build(self, **kwargs):
        if(self.type == 'qubit'):
            self.Ej = kwargs["Ej"]
            self.Ec = kwargs["Ec"]
            self.ng = kwargs["ng"]
            self.ncut = kwargs["ncut"]
            scq.Transmon(self.Ej,self.Ec,self.ng, self.ncut)
        if(self.type == 'cavity' or self.type == 'oscillator'):
            self.E_osc = kwargs["E_osc"]
            self.dim = kwargs["dim"]
            scq.Oscillator(self.E_osc, truncated_dim=self.dim)
    
