import numpy as np
from qutip import Options, ket2dm
import qutip as qt
import matplotlib.pyplot as plt
from itertools import product
from tqdm.notebook import tqdm
import cmath
from qutip.qip.operations import iswap
from scipy.optimize import curve_fit


class Time_functions():
    def __init__(self, w1, w2, w3, w4, wp):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.wp = wp

    def functions(self):
        # build the pulses for the unitary 
        def int_func(w1,w2,wp,t):
            a=((np.exp(-1j*(w1-w2+wp)*t))/(-1j*(w1-w2+wp)) - (1/(-1j*(w1-w2+wp))))
            return a

        def int_func_conj_wp(w1,w2,wp,t):
            a = ((np.exp(-1j*(w1-w2-wp)*t))/(-1j*(w1-w2-wp)) - (1/(-1j*(w1-w2-wp))))
            return a

        def int_func_conj(w1,w2,wp,t):
            a = ((np.exp(1j*(w1-w2+wp)*t))/(1j*(w1-w2+wp)) - (1/(1j*(w1-w2+wp))))
            return a 

        def int_func_conj_wp_conj(w1,w2,wp,t):
            a = ((np.exp(1j*(w1-w2-wp)*t))/(1j*(w1-w2-wp)) - (1/(1j*(w1-w2-wp))))
            return a

        T = 100

        qubit1_qubit2_adj_val = int_func_conj_wp(self.w1,self.w2,self.wp,T)
        qubit1_adj_qubit2_val = int_func_conj_wp_conj(self.w1,self.w2,self.wp,T)
        qubit1_qubit3_adj_val = int_func(self.w1,self.w3,self.wp,T) + int_func_conj_wp(self.w1,self.w3,self.wp,T)
        qubit2_qubit3_adj_val = int_func(self.w2,self.w3,self.wp,T) + int_func_conj_wp(self.w2,self.w3,self.wp,T)
        qubit1_adj_qubit3_val = int_func(self.w3,self.w1,self.wp,T) + int_func_conj_wp(self.w3,self.w1,self.wp,T)
        qubit2_adj_qubit3_val = int_func(self.w3,self.w2,self.wp,T) + int_func_conj_wp(self.w3,self.w2,self.wp,T)
        qubit1_qubit4_adj_val = int_func(self.w1,self.w4,self.wp,T) + int_func_conj_wp(self.w1,self.w4,self.wp,T)
        qubit2_qubit4_adj_val = int_func(self.w2,self.w4,self.wp,T) + int_func_conj_wp(self.w2,self.w4,self.wp,T)
        qubit1_adj_qubit4_val = int_func(self.w4,self.w1,self.wp,T) + int_func_conj_wp(self.w4,self.w1,self.wp,T)
        qubit2_adj_qubit4_val = int_func(self.w4,self.w2,self.wp,T) + int_func_conj_wp(self.w4,self.w2,self.wp,T)
        qubit3_qubit4_adj_val = int_func(self.w3,self.w4,self.wp,T) + int_func_conj_wp(self.w3,self.w4,self.wp,T)
        qubit3_adj_qubit4_val = int_func(self.w4,self.w3,self.wp,T) + int_func_conj_wp(self.w4,self.w3,self.wp,T)

        # building the time_multiplier list 
        T_mult = [
        T,
        qubit1_qubit2_adj_val,
        qubit1_adj_qubit2_val,
        qubit2_adj_qubit3_val,
        qubit2_qubit3_adj_val,
        qubit2_adj_qubit4_val,
        qubit2_qubit4_adj_val,
        qubit1_adj_qubit3_val,
        qubit1_qubit3_adj_val,
        qubit1_adj_qubit4_val,
        qubit1_qubit4_adj_val,
        qubit3_qubit4_adj_val,
        qubit3_adj_qubit4_val
        ]

        return T_mult