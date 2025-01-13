import qutip as qt
from qutip import tensor, basis
import numpy as np
from scipy.linalg import fractional_matrix_power


class Trotterization():
    def __init__(self, trotter_dt, T1, T2, dim, num_qubits, qudit):
        self.trotter_dt = trotter_dt
        self.T1 = T1
        self.T2 = T2
        self.dim = dim
        self.num_qubits = num_qubits
        self.qudit = qudit
        

        # take the state that you have and actually trotterize it to make it make sense 
        return None
    
    def apply(self, rho, duration, unitary, errors):
        # from quantum_logical.loss_channels import Loss_channel
        from quantum_logical.kraus_operators import Kraus_operators



        # pulling in the error channels 
        if self.qudit == "qubit":
            a = Kraus_operators(trotter_dt=self.trotter_dt / len(unitary), T1=self.T1, T2=self.T2, num_qubits=self.num_qubits, dim=self.dim, qudits=self.qudit)

            self.error1, self.error2 = a.channel()
            # look into how many time steps are necessary for trotterization 
            num_steps = int(duration / self.trotter_dt)

            # create the fractional unitary 
            unitary_frac = [fractional_matrix_power(i, 1 / num_steps) for i in unitary]


            state = rho
            states = []
            for _ in range(num_steps):
                for i in unitary_frac:
                    if errors == True:
                        state = sum([ops * state * ops.dag() for ops in self.error1])
                        state = state / state.tr()
                        state = sum([ops * state * ops.dag() for ops in self.error2])
                        state = state / state.tr()
                # allows multiple unitaries to be trotterized simultaneously 
                    state = qt.Qobj(i, dims=rho.dims) * state * qt.Qobj(i, dims=rho.dims).dag()
                    state = state / state.tr()
                states.append(state)
        elif self.qudit == "qutrit":
            a = Kraus_operators(trotter_dt=self.trotter_dt / len(unitary), T1=self.T1, T2=self.T2, num_qubits=self.num_qubits, dim=self.dim, qudits=self.qudit)
            self.error1, self.error2 = a.channel_qutrit()
    
            # look into how many time steps are necessary for trotterization 
            num_steps = int(duration / self.trotter_dt)

            # create the fractional untiary 
            unitary_frac = [fractional_matrix_power(i, 1 / num_steps) for i in unitary]

            state = rho
            states = []
            for _ in range(num_steps):
                for i in unitary_frac:
                    if errors == True:
                        state = sum([ops * state * ops.dag() for ops in self.error1])
                        state = state / state.tr()
                        state = sum([ops * state * ops.dag() for ops in self.error2])
                        state = state / state.tr()
                # allows multiple unitaries to be trotterized simultaneously 
                    state = qt.Qobj(i, dims=rho.dims) * state * qt.Qobj(i, dims=rho.dims).dag()
                    state = state / state.tr()
                states.append(state)
        elif self.qudit == "ququart":
            a = Kraus_operators(trotter_dt=self.trotter_dt / len(unitary), T1=self.T1, T2=self.T2, num_qubits=self.num_qubits, dim=self.dim, qudits=self.qudit)
            self.error1, self.error2 = a.channel_ququart()
    
            # look into how many time steps are necessary for trotterization 
            num_steps = int(duration / self.trotter_dt)

            # create the fractional untiary 
            unitary_frac = [fractional_matrix_power(i, 1 / num_steps) for i in unitary]

            state = rho
            states = []
            for _ in range(num_steps):
                for i in unitary_frac:
                    if errors == True:
                        state = sum([ops * state * ops.dag() for ops in self.error1])
                        state = state / state.tr()
                        state = sum([ops * state * ops.dag() for ops in self.error2])
                        state = state / state.tr()
                # allows multiple unitaries to be trotterized simultaneously 
                    state = qt.Qobj(i, dims=rho.dims) * state * qt.Qobj(i, dims=rho.dims).dag()
                    state = state / state.tr()
                states.append(state)

        
        return states
    
    def print_error(self):
        from quantum_logical.kraus_operators import Kraus_operators

        a = Kraus_operators(trotter_dt=self.trotter_dt, T1=self.T1, T2=self.T2, num_qubits=self.num_qubits, dim=self.dim, qudits=self.qudit)
        self.error1, self.error2 = a.channel()

        return self.error1, self.error2
    def print_qutrit(self):
        from quantum_logical.kraus_operators import Kraus_operators

        a = Kraus_operators(trotter_dt=self.trotter_dt, T1=self.T1, T2=self.T2, num_qubits=self.num_qubits, dim=self.dim, qudits=self.qudit)
        self.error1, self.error2 = a.channel_qutrit()
    
        return self.error1, self.error2
    
    def print_ququart(self):
        from quantum_logical.kraus_operators import Kraus_operators

        a = Kraus_operators(trotter_dt=self.trotter_dt, T1=self.T1, T2=self.T2, num_qubits=self.num_qubits, dim=self.dim, qudits=self.qudit)
        self.error1, self.error2 = a.channel_ququart()
    
        return self.error1, self.error2