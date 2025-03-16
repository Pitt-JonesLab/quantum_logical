from qutip import Qobj


class Trotterization():
    def __init__(self, trotter_dt, N, T1, T2, qudit, dim):
        self.trotter_dt = trotter_dt
        self.num_qubits = N
        self.T1 = T1
        self.T2 = T2
        self.qudit = qudit
        self.dim = dim

        return None

    def unitary_frac(self, unitary, slice_):
        from scipy.linalg import logm, expm
        from scipy import constants

        h_bar = constants.hbar  # useful constant

        hamiltonian = 1j * logm(unitary) * h_bar  # building hamiltonian

        U_frac = expm((1j * hamiltonian * (1/slice_)) / h_bar)  # building unitary

        return U_frac

    def apply(self, rho, duration, unitary, errors, amp=True, dephasing=True):

        self.unitary = unitary
        from quantum_logical.kraus_operators import Kraus_operators

        # Choose error channels based on qudit type
        if self.qudit == "qubit":
            a = Kraus_operators(trotter_dt=self.trotter_dt / len(unitary),
                                T1=self.T1, T2=self.T2,
                                num_qubits=self.num_qubits, dim=self.dim,
                                qudits=self.qudit)
            self.error1, self.error2 = a.channel()

        elif self.qudit == "qutrit":
            a = Kraus_operators(trotter_dt=self.trotter_dt / len(unitary),
                                T1=self.T1, T2=self.T2,
                                num_qubits=self.num_qubits, dim=self.dim,
                                qudits=self.qudit)
            self.error1, self.error2 = a.channel_qutrit()

        elif self.qudit == "ququart":
            a = Kraus_operators(trotter_dt=self.trotter_dt / len(unitary),
                                T1=self.T1, T2=self.T2,
                                num_qubits=self.num_qubits, dim=self.dim,
                                qudits=self.qudit)
            self.error1, self.error2 = a.channel_ququart()

        num_steps = int(duration / self.trotter_dt)

        # Convert unitary to fractional powers
        unitary_fr = [self.unitary_frac(i.full(), num_steps)
                        for i in unitary]

        state = rho
        states = []

        # Loop over time steps
        for _ in range(num_steps):
            for i in unitary_fr:
                if errors:  # Apply error channels if errors=True
                    if amp:
                        state = sum([ops * state * ops.dag()
                                     for ops in self.error1])
                    if dephasing:
                        state = sum([ops * state * ops.dag()
                                     for ops in self.error2])
                # Apply unitary
                state = Qobj(i, dims=rho.dims) * state * Qobj(i, dims=rho.dims).dag()
                state = state / state.tr()

            states.append(state)

        return states
