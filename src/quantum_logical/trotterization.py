import qutip as qt
from qutip import tensor, basis
import numpy as np
from scipy.linalg import fractional_matrix_power


class Trotterization():
    def __init__(self, trotter_dt, T1, T2, dim, num_qubits):
        self.trotter_dt = trotter_dt
        self.T1 = T1
        self.T2 = T2
        self.dim = dim
        self.num_qubits = num_qubits
        


        # take the state that you have and actually trotterize it to make it make sense 
        return None
    
    def apply(self, rho, duration, unitary):
        from quantum_logical.loss_channels import Loss_channel



        # pulling in the error channels 
        a = Loss_channel(trotter_dt=self.trotter_dt, T1=self.T1, T2=self.T2, num_qubits=self.num_qubits, dim=self.dim)
        self.error1, self.error2 = a.channel()
    
        # look into how many time steps are necessary for trotterization 
        num_steps = int(duration / self.trotter_dt)


        # create the fractional untiary 
        unitary_frac = fractional_matrix_power(unitary, 1 / num_steps)

        state = rho
        states = []
        for _ in range(num_steps):
            state = sum([ops * state * ops.dag() for ops in self.error1])
            state = state / state.tr()
            state = sum([ops * state * ops.dag() for ops in self.error2])
            state = state / state.tr()
            state = qt.Qobj(unitary_frac, dims=rho.dims) * state * qt.Qobj(unitary_frac, dims=rho.dims).dag()
            states.append(state)

        
        return states
    
    def print_error(self):
        from quantum_logical.loss_channels import Loss_channel


        a = Loss_channel(trotter_dt=self.trotter_dt, T1=self.T1, T2=self.T2, num_qubits=self.num_qubits, dim=self.dim)
        self.error1, self.error2 = a.channel()

        # return self.error1
        return self.error1, self.error2