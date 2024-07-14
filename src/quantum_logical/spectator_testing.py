import numpy as np
from qutip import Options, ket2dm
import qutip as qt
from quantum_logical.pulsesim import QuantumSystem, Pulse
from quantum_logical.pulsesim.mode import QubitMode, SNAILMode, CavityMode
from quantum_logical.pulsesim.build_hamiltonian import Build_hamiltonian
import matplotlib.pyplot as plt
from itertools import product
from tqdm.notebook import tqdm
import cmath
from qutip.qip.operations import iswap
from scipy.optimize import curve_fit

class spectator():
    def __init__(self, qubit_3, lam_power):
        self.qubit_3 = qubit_3
        self.lam_power = lam_power

        lambda_power = []

        # build the four qubit system

        w1_un = 4
        w2_un = 6
        # w3_un = 4.00000000000001
        # w4_un = 5.99999999999999
        w3_un = qubit_3
        # w4_un = 5.5
        ws_un = 6 - (1 / 3) * (1 / 2)

        self.w1 = w1_un
        self.w2 = w2_un

        dim = 2
        qubit1 = QubitMode(mode_type = "Qubit",
            name="q1", dim=dim, freq=w1_un, alpha=-0.161, T1=1e2, T2=5e1
        )
        qubit2 = QubitMode(mode_type = "Qubit",
            name="q2", dim=dim, freq=w2_un, alpha=-0.1275, T1=1e2, T2=5e1
        )
        qubit3 = QubitMode(mode_type = "Qubit",
            name="q3", dim=dim, freq=w3_un, alpha=-0.160, T1=1e2, T2=5e1
        )
        # qubit4 = QubitMode(mode_type = "Qubit",
        #     name="q4", dim=dim, freq=w4_un, alpha=-0.159, T1=1e2, T2=5e1
        # )
        # qubits = [qubit1, qubit2, qubit3, qubit4]
        qubits = [qubit1, qubit2, qubit3]
        snail = SNAILMode(mode_type = "Snail", name="s", freq=ws_un, g3=0.3, dim=10, T1=1e3, T2=5e2)
        _couplings = {
            frozenset([qubit1, snail]): 2 * np.pi * 0.05467,
            frozenset([qubit2, snail]): 2 * np.pi * 0.0435,
            frozenset([qubit3, snail]): 2 * np.pi * 0.04875,
        }

        qs = QuantumSystem(qubits + [snail], couplings=_couplings)

        # important multipliers and hamiltonian prefactors 
        l1 = l2 = l3 = .1

        w1 = qubit1.freq / (2 * np.pi) 
        w2 = qubit2.freq / (2 * np.pi) 
        w3 = qubit3.freq / (2 * np.pi)  
        # w4 = qubit4.freq / (2 * np.pi) 
        ws = snail.freq / (2 * np.pi)

        # time over which to optimize
        T = 100

        #pump(drive) frequency
        # wp = w1 - w2
        wp = w2 - w1

        # # unchanged terms of hamiltonian
        H_no_time = 6*(l1**2)*(qs.modes_a[qubit1]*qs.modes_a_dag[qubit2] + qs.modes_a[qubit2]*qs.modes_a_dag[qubit1])

        # using the class created to build the hamiltonian 
        # first and second are involved in the driving and the third and the fourth are the spectating ones
        # Hs = Build_hamiltonian(l1, qs, qubit1, qubit2, qubit3, qubit4)
        # H_main_qubits = Hs.build_drive_hamiltonian()

        #terms that come from the gate that is desired that do not go to one
        qubit1_qubit2_adj_H = 6*(l1**2)*qs.modes_a[qubit1]*qs.modes_a_dag[qubit2]
        qubit1_adj_qubit2_H = 6*(l1**2)*qs.modes_a[qubit2]*qs.modes_a_dag[qubit1]

        H_main_qubits = [
        H_no_time,
        qubit1_qubit2_adj_H,
        qubit1_adj_qubit2_H
        ]

        # building the added terms 
        # H_added = Hs.build_drive_hamiltonian()

        # # cell that will determine over which lambda power to evaluate

        qubit3_qubit2_adj_H = qs.modes_a[qubit3]*qs.modes_a_dag[qubit2]
        qubit3_adj_qubit2_H = qs.modes_a[qubit2]*qs.modes_a_dag[qubit3]
        # qubit4_qubit2_adj_H = qs.modes_a[qubit4]*qs.modes_a_dag[qubit2]
        # qubit4_adj_qubit2_H = qs.modes_a[qubit2]*qs.modes_a_dag[qubit4]
        qubit3_qubit1_adj_H = qs.modes_a[qubit3]*qs.modes_a_dag[qubit1]
        qubit3_adj_qubit1_H = qs.modes_a[qubit1]*qs.modes_a_dag[qubit3]
        # qubit4_qubit1_adj_H = qs.modes_a[qubit4]*qs.modes_a_dag[qubit1]
        # qubit4_adj_qubit1_H = qs.modes_a[qubit1]*qs.modes_a_dag[qubit4]
        # qubit3_qubit4_adj_H = qs.modes_a[qubit3]*qs.modes_a_dag[qubit4]
        # qubit3_adj_qubit4_H = qs.modes_a[qubit4]*qs.modes_a_dag[qubit3]

        # H_added = [
        #         qubit3_qubit2_adj_H,
        #         qubit3_adj_qubit2_H,
        #         qubit4_qubit2_adj_H,
        #         qubit4_adj_qubit2_H,
        #         qubit3_qubit1_adj_H,
        #         qubit3_adj_qubit1_H,
        #         qubit4_qubit1_adj_H,
        #         qubit4_adj_qubit1_H, 
        #         qubit3_qubit4_adj_H,
        #         qubit3_adj_qubit4_H
        #         ]

        H_added = [
                qubit3_qubit2_adj_H,
                qubit3_adj_qubit2_H,
                qubit3_qubit1_adj_H,
                qubit3_adj_qubit1_H
                ]


        H_modified = []
        def choose_lambda(choice):
            for i in range(len(H_added)):
                if(choice == 2):
                    H_modified.append(6 * (l1**2) * H_added[i])
                elif(choice == 3):
                    H_modified.append(6 * (l1**3) * H_added[i])
                elif(choice == 4):
                    H_modified.append(6 * (l1**4) * H_added[i])
                elif(choice == 5):
                    H_modified.append(6 * (l1**5) * H_added[i])
                elif(choice == 6):
                    H_modified.append(6 * (l1**6) * H_added[i])

            return H_modified

        choice = self.lam_power
        H_total = []
        H_total.extend(H_main_qubits)
        # call in the function 
        H_mod = choose_lambda(choice=choice)
        lambda_power.append(choice)
        H_total.extend(H_mod)
        len(H_total)

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

        qubit1_qubit2_adj_val = int_func_conj_wp(w1,w2,wp,T)
        qubit1_adj_qubit2_val = int_func_conj_wp_conj(w1,w2,wp,T)
        qubit1_qubit3_adj_val = int_func(w1,w3,wp,T) + int_func_conj_wp(w1,w3,wp,T)
        qubit2_qubit3_adj_val = int_func(w2,w3,wp,T) + int_func_conj_wp(w2,w3,wp,T)
        qubit1_adj_qubit3_val = int_func(w3,w1,wp,T) + int_func_conj_wp(w3,w1,wp,T)
        qubit2_adj_qubit3_val = int_func(w3,w2,wp,T) + int_func_conj_wp(w3,w2,wp,T)
        # qubit1_qubit4_adj_val = int_func(w1,w4,wp,T) + int_func_conj_wp(w1,w4,wp,T)
        # qubit2_qubit4_adj_val = int_func(w2,w4,wp,T) + int_func_conj_wp(w2,w4,wp,T)
        # qubit1_adj_qubit4_val = int_func(w4,w1,wp,T) + int_func_conj_wp(w4,w1,wp,T)
        # qubit2_adj_qubit4_val = int_func(w4,w2,wp,T) + int_func_conj_wp(w4,w2,wp,T)
        # qubit3_qubit4_adj_val = int_func(w3,w4,wp,T) + int_func_conj_wp(w3,w4,wp,T)
        # qubit3_adj_qubit4_val = int_func(w4,w3,wp,T) + int_func_conj_wp(w4,w3,wp,T)

        # building the time_multiplier list 
        # T_mult = [
        # T,
        # qubit1_qubit2_adj_val,
        # qubit1_adj_qubit2_val,
        # qubit2_adj_qubit3_val,
        # qubit2_qubit3_adj_val,
        # qubit2_adj_qubit4_val,
        # qubit2_qubit4_adj_val,
        # qubit1_adj_qubit3_val,
        # qubit1_qubit3_adj_val,
        # qubit1_adj_qubit4_val,
        # qubit1_qubit4_adj_val,
        # qubit3_qubit4_adj_val,
        # qubit3_adj_qubit4_val
        # ]

        T_mult = [
        T,
        qubit1_qubit2_adj_val,
        qubit1_adj_qubit2_val,
        qubit2_adj_qubit3_val,
        qubit2_qubit3_adj_val,
        qubit1_adj_qubit3_val,
        qubit1_qubit3_adj_val
        ]


        def mod_amount(count):
            H_tot = []
            ts = []
            ts.append(T_mult[0])
            ts.append(T_mult[1])
            ts.append(T_mult[2])
            H_tot.extend(H_main_qubits)
            for _ in range(count):
                for i in range(3, len(H_total)):
                    H_tot.append(H_total[i])
                    ts.append(T_mult[i])

            return [H_tot, ts]

        num_module = 1
        res = mod_amount(num_module)

        # extract out the hamiltonian and the time multiplier from the mod count results 
        multiplier_times = res[1]
        H = res[0]

        # build the desired unitary 
        # U_targ = U = qt.tensor(qt.qip.operations.iswap(N=2),qt.qeye(2),qt.identity(cavity.dim))
        desired_U = iswap()  # The iSWAP gate for a 2-qubit system

        # Create isometries for qubit 1 and qubit 2 to extend the {g, e} subspace action to the full qubit space
        identity_isometry = (
            qt.basis(dim, 0) * qt.basis(2, 0).dag()
            + qt.basis(dim, 1) * qt.basis(2, 1).dag()
        )
        identity_isometry = qt.tensor(identity_isometry, identity_isometry)

        # Apply the isometry to extend the gate action
        extended_q1_q2 = identity_isometry * desired_U * identity_isometry.dag()

        # Tensor with identity matrices for the remaining qubits and the SNAIL mode
        for mode in qs.modes[2:]:  # Skip the first two qubits as they're already included
            extended_q1_q2 = qt.tensor(extended_q1_q2, qt.qeye(mode.dim))

        # The extended_iswap_q1_q2 now acts as the desired iSWAP gate on {g, e} of qubits 1 and 2, and as identity on the rest
        desired_U = extended_q1_q2

        # build the designed unitary
        # run the fidelity analysis over the expected gate 
        amps = np.linspace(0, 2, 300)
        results = []
        fids = []
        i_count = []

        # non-optimization cell 
        eta = ((2 * wp) / ((wp**2) - (ws**2))) * 1.9665551839464883
        Z = []
        Z.clear()
        for j in range(len(H)):
            
            Z.append(eta * H[j] * multiplier_times[j])

        w = sum(Z)
        U_propagator = (-1j * w).expm()

        #calculate the fidelity
        self.fid = np.abs(qt.average_gate_fidelity(desired_U, U_propagator))
        # results.append([i, fid])
        # fids.append(fid)
        # i_count.append(i)

        
        return None

    def results(self):
        return self.fid
    
    def min_gate_sep(self):
        gate12 = np.abs(self.w1 - self.w2)
        gate13 = np.abs(self.w1 - self.qubit_3)
        gate23 = np.abs(self.w2 - self.qubit_3)

        sep12_13 = np.abs(gate12 - gate13)
        sep12_23 = np.abs(gate12 - gate23)
        sep13_23 = np.abs(gate13 - gate23)

        seps = [sep12_13, sep12_23, sep13_23]

        return min(seps)


