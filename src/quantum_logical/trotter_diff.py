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
    
    def apply(self, rho, duration, unitary, errors, amp=True, dephasing=True):
        from quantum_logical.kraus_operators import Kraus_operators

        # Choose error channels based on qudit type
        if self.qudit == "qubit":
            a = Kraus_operators(trotter_dt=self.trotter_dt / len(unitary),
                                T1=self.T1, T2=self.T2, num_qubits=self.num_qubits,
                                  dim=self.dim, qudits=self.qudit)
            self.error1, self.error2 = a.channel()
        
        elif self.qudit == "qutrit":
            a = Kraus_operators(trotter_dt=self.trotter_dt / len(unitary),
                                T1=self.T1, T2=self.T2, num_qubits=self.num_qubits,
                                  dim=self.dim, qudits=self.qudit)
            self.error1, self.error2 = a.channel_qutrit()
        
        elif self.qudit == "ququart":
            a = Kraus_operators(trotter_dt=self.trotter_dt / len(unitary), 
                                T1=self.T1, T2=self.T2, num_qubits=self.num_qubits,
                                  dim=self.dim, qudits=self.qudit)
            self.error1, self.error2 = a.channel_ququart()

        num_steps = int(duration / self.trotter_dt)

        # Convert unitary to fractional powers
        unitary_frac = [fractional_matrix_power(i.full(), 1 / num_steps) for i in unitary]

        state = rho
        states = []

                            
        # Loop over time steps
        for _ in range(num_steps):
            for i in unitary_frac:
                if errors:  # Apply error channels if errors=True
                    if amp:
                        state = sum([ops * state * ops.dag() for ops in self.error1])
                    if dephasing:
                        state = sum([ops * state * ops.dag() for ops in self.error2])
                # Apply unitary
                state = qt.Qobj(i, dims=rho.dims) * state * qt.Qobj(i, dims=rho.dims).dag()
                state = state / state.tr()

            states.append(state)


        return states

