import numpy as np
from itertools import product
import qutip as qt
import h5py 
from quantum_logical.pulsesim import QuantumSystem, Pulse
from quantum_logical.pulsesim.mode import QubitMode, SNAILMode, CavityMode
from quantum_logical.pulsesim.build_hamiltonian import Build_hamiltonian
import matplotlib.pyplot as plt
from qutip.qip.operations import iswap

class Work_flow():
    def __init__(self, num_qubits, lamb, freq_list, pumped_qubits):
        self.qubit1_freq = freq_list[0]
        self.qubit2_freq = freq_list[1]
        self.qubit3_freq = freq_list[2]
        self.qubit4_freq = freq_list[3]
        self.freq_list = freq_list
        self.pumped_qubits = pumped_qubits
        self.num_qubits = num_qubits
        self._lambda = lamb

        return None

    def calibrator(self):

        w1 = self.qubit1_freq 
        w2 = self.qubit2_freq
        w3 = self.qubit3_freq
        w4 = self.qubit4_freq


        # generalize these next few lines
        dim = 2
        Qubit_modes = []
    
        # for i in range(self.num_qubits):
        #     Qubit_modes.append(QubitMode(mode_type = "Qubit",
        #     name=f"q{i}", dim=dim, freq=self.freq_list[i], alpha=-0.161, T1=1e2, T2=5e1
        # ))





        qubit1 = QubitMode(mode_type = "Qubit",
            name="q1", dim=dim, freq=self.qubit1_freq, alpha=-0.161, T1=1e2, T2=5e1
        )
        qubit2 = QubitMode(mode_type = "Qubit",
            name="q2", dim=dim, freq=self.qubit2_freq, alpha=-0.1275, T1=1e2, T2=5e1
        )
        qubit3 = QubitMode(mode_type = "Qubit",
            name="q3", dim=dim, freq=self.qubit3_freq, alpha=-0.160, T1=1e2, T2=5e1
        )
        qubit4 = QubitMode(mode_type = "Qubit",
            name="q4", dim=dim, freq=self.qubit4_freq, alpha=-0.159, T1=1e2, T2=5e1
        )
        qubits = [qubit1, qubit2, qubit3, qubit4]

        ws = w1 + ((w1 - w2) / 3)
        # ws = 6 - (1 / 3) * (1 / 2)

        snail = SNAILMode(mode_type = "Snail", name="s", freq=ws, g3=0.3, dim=10, T1=1e3, T2=5e2)
        _couplings = {
            frozenset([qubit1, snail]): 2 * np.pi * 0.05467,
            frozenset([qubit2, snail]): 2 * np.pi * 0.0435,
            frozenset([qubit3, snail]): 2 * np.pi * 0.04875,
        }

        qs = QuantumSystem(qubits + [snail], couplings=_couplings)
        # time over which to optimize
        T = 100

        g_3 = .3 # this should be given above 

        #pump(drive) frequency
        # this should be made more specific to different qubits 
        # this statement above may introduce a complexity that is not necessary 
        # wp = w1 - w2
        wp = np.abs(w1 - w2)

        hamiltonian_coeff = 6 * (self._lambda ** 2) * g_3 # check this line against the crowding code because they should spit out the same thing and they do not 

        # # unchanged terms of hamiltonian
        H_no_time = hamiltonian_coeff * (qs.modes_a[qubit1]*qs.modes_a_dag[qubit2] + qs.modes_a[qubit2]*qs.modes_a_dag[qubit1])

        #terms that come from the gate that is desired that do not go to one
        qubit1_qubit2_adj_H = hamiltonian_coeff * qs.modes_a[qubit1]*qs.modes_a_dag[qubit2]
        qubit1_adj_qubit2_H = hamiltonian_coeff * qs.modes_a[qubit2]*qs.modes_a_dag[qubit1]

        H_main_qubits = [
        H_no_time,
        qubit1_qubit2_adj_H,
        qubit1_adj_qubit2_H
        ]

        # qubit3_qubit2_adj_H = qs.modes_a[qubit3]*qs.modes_a_dag[qubit2]
        # qubit3_adj_qubit2_H = qs.modes_a[qubit2]*qs.modes_a_dag[qubit3]
        # qubit4_qubit2_adj_H = qs.modes_a[qubit4]*qs.modes_a_dag[qubit2]
        # qubit4_adj_qubit2_H = qs.modes_a[qubit2]*qs.modes_a_dag[qubit4]
        # qubit3_qubit1_adj_H = qs.modes_a[qubit3]*qs.modes_a_dag[qubit1]
        # qubit3_adj_qubit1_H = qs.modes_a[qubit1]*qs.modes_a_dag[qubit3]
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
        
        # create the added gates in a more generalized manner
        # for i in range(self.num_qubits):

        
        # def choose_lambda(choice):
        #     # for i in range(len(H_added)):
        #     if(choice == 2):
        #         # H_modified.append(6 * (l1**2) * H_added[i])
        #         H_modified = [
        #             hamiltonian_coeff * qubit3_qubit2_adj_H,
        #             hamiltonian_coeff * qubit3_adj_qubit2_H,
        #             hamiltonian_coeff * qubit4_qubit2_adj_H,
        #             hamiltonian_coeff * qubit4_adj_qubit2_H,
        #             hamiltonian_coeff * qubit3_qubit1_adj_H,
        #             hamiltonian_coeff * qubit3_adj_qubit1_H,
        #             hamiltonian_coeff * qubit4_qubit1_adj_H,
        #             hamiltonian_coeff * qubit4_adj_qubit1_H,
        #             hamiltonian_coeff * qubit3_qubit4_adj_H,
        #             hamiltonian_coeff * qubit3_adj_qubit4_H
        #             ]
                    # H_modified= [6 * (l1**2) *qubit3_qubit4_adj_H,
                    # 6 * (l1**2) *qubit3_adj_qubit4_H]
                # elif(choice == 3):
                    # H_modified.append(hamiltonian_coeff * H_added[i])
                # elif(choice == 4):
                #     # H_modified.append(6 * (l1**4) * H_added[i])
                #     H_modified = [
                #         hamiltonian_coeff * qubit3_qubit2_adj_H,
                #         hamiltonian_coeff * qubit3_adj_qubit2_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit4_qubit2_adj_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit4_adj_qubit2_H,
                #         hamiltonian_coeff * qubit3_qubit1_adj_H,
                #         hamiltonian_coeff * qubit3_adj_qubit1_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit4_qubit1_adj_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit4_adj_qubit1_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit3_qubit4_adj_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit3_adj_qubit4_H
                #         ]
                #     # H_modified= [6 * (l1**4) *qubit3_qubit4_adj_H,
                #     # 6 * (l1**4) *qubit3_adj_qubit4_H]
                # # elif(choice == 5):
                #     # H_modified.append(hamiltonian_coeff * H_added[i])
                # elif(choice == 6):
                #     # H_modified.append(6 * (l1**6) * H_added[i])
                #     H_modified = [
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit3_qubit2_adj_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit3_adj_qubit2_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit4_qubit2_adj_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit4_adj_qubit2_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit3_qubit1_adj_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit3_adj_qubit1_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit4_qubit1_adj_H,
                #         hamiltonian_coeff * (self._lambda ** 2) * qubit4_adj_qubit1_H,
                #         hamiltonian_coeff * (self._lambda ** 4) * qubit3_qubit4_adj_H,
                #         hamiltonian_coeff * (self._lambda ** 4) * qubit3_adj_qubit4_H
                #         ]
                #     # H_modified= [6 * (l1**6) *qubit3_qubit4_adj_H,
                #     # 6 * (l1**6) *qubit3_adj_qubit4_H]

            # return H_modified

        
        # choice = 2 # this is a place holder that needs updated
        H_total = []
        H_total.extend(H_main_qubits)
        # call in the function 
        # H_mod = choose_lambda(choice=choice)
        # lambda_power.append(choice)
        # H_total.extend(H_mod)
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
        qubit1_qubit4_adj_val = int_func(w1,w4,wp,T) + int_func_conj_wp(w1,w4,wp,T)
        qubit2_qubit4_adj_val = int_func(w2,w4,wp,T) + int_func_conj_wp(w2,w4,wp,T)
        qubit1_adj_qubit4_val = int_func(w4,w1,wp,T) + int_func_conj_wp(w4,w1,wp,T)
        qubit2_adj_qubit4_val = int_func(w4,w2,wp,T) + int_func_conj_wp(w4,w2,wp,T)
        qubit3_qubit4_adj_val = int_func(w3,w4,wp,T) + int_func_conj_wp(w3,w4,wp,T)
        qubit3_adj_qubit4_val = int_func(w4,w3,wp,T) + int_func_conj_wp(w4,w3,wp,T)

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

        num_module = 0 # this is a placeholder this will be specific to the topology selected (this is the point at which to calibrate the gate)
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
        # self.amp = T * hamiltonian_coeff * (np.pi / 2)
        ampi = 3
        amps = np.linspace(0, ampi, 300)
        fids = []
        eta_list = []


        # the issue here is that it is double optimizing which you do not want to do 
        # this means that you have to calibrate the gate first to get what the amp should be and then keep it fixed for all of the frequency tests and this should give a good result 
        # non-optimization cell 
        for amp in amps:
            eta = ((2 * wp) / ((wp**2) - (ws**2))) * amp
            Z = []
            Z.clear()
            for j in range(len(H)):
                
                Z.append(eta * H[j] * multiplier_times[j])

            w = sum(Z)
            U_propagator = (-1j * w).expm()

            #calculate the fidelity
            fid = np.abs(qt.average_gate_fidelity(desired_U, U_propagator))
            # results.append([i, fid])
            fids.append(fid)
            eta_list.append(eta)
            # i_count.append(i)
        max_fid = max(fids)
        eta_max = eta_list[fids.index(max_fid)]

        return [max_fid, eta_max]
    

    # THIS IS THE ACTUAL EXPERIMENT FUNCTION
    # 
    # 
    def Work_experiment(self, eta):
        w1 = self.qubit1_freq 
        w2 = self.qubit2_freq
        w3 = self.qubit3_freq 
        w4 = self.qubit4_freq
        # frequency_list = [w1, w2]
        # for i in range(len(frequencies)):
        #     frequency_list.append(frequencies[i])


        # generalize these next few lines
        dim = 2
        # Qubit_modes = []
    
        # for i in range(self.num_qubits):
        #     Qubit_modes.append(QubitMode(mode_type = "Qubit",
        #     name=f"q{i}", dim=dim, freq=self.freq_list[i], alpha=-0.161, T1=1e2, T2=5e1
        # ))





        qubit1 = QubitMode(mode_type = "Qubit",
            name="q1", dim=dim, freq=self.qubit1_freq, alpha=-0.161, T1=1e2, T2=5e1
        )
        qubit2 = QubitMode(mode_type = "Qubit",
            name="q2", dim=dim, freq=self.qubit2_freq, alpha=-0.1275, T1=1e2, T2=5e1
        )
        qubit3 = QubitMode(mode_type = "Qubit",
            name="q3", dim=dim, freq=self.qubit3_freq, alpha=-0.160, T1=1e2, T2=5e1
        )
        qubit4 = QubitMode(mode_type = "Qubit",
            name="q4", dim=dim, freq=self.qubit4_freq, alpha=-0.159, T1=1e2, T2=5e1
        )
        qubits = [qubit1, qubit2, qubit3, qubit4]

        ws = w1 + ((w1 - w2) / 3)
        # ws = 6 - (1 / 3) * (1 / 2)

        snail = SNAILMode(mode_type = "Snail", name="s", freq=ws, g3=0.3, dim=10, T1=1e3, T2=5e2)
        _couplings = {
            frozenset([qubit1, snail]): 2 * np.pi * 0.05467,
            frozenset([qubit2, snail]): 2 * np.pi * 0.0435,
            frozenset([qubit3, snail]): 2 * np.pi * 0.04875,
        }

        qs = QuantumSystem(qubits + [snail], couplings=_couplings)
        # time over which to optimize
        T = 100

        g_3 = .3 # this should be given above 

        #pump(drive) frequency
        # this should be made more specific to different qubits 
        # this statement above may introduce a complexity that is not necessary 
        # wp = w1 - w2
        wp = np.abs(w1 - w2)

        hamiltonian_coeff = 6 * (self._lambda ** 2) * g_3 # check this line against the crowding code because they should spit out the same thing and they do not 

        # # unchanged terms of hamiltonian
        H_no_time = hamiltonian_coeff * (qs.modes_a[qubit1]*qs.modes_a_dag[qubit2] + qs.modes_a[qubit2]*qs.modes_a_dag[qubit1])

        #terms that come from the gate that is desired that do not go to one
        qubit1_qubit2_adj_H = hamiltonian_coeff * qs.modes_a[qubit1]*qs.modes_a_dag[qubit2]
        qubit1_adj_qubit2_H = hamiltonian_coeff * qs.modes_a[qubit2]*qs.modes_a_dag[qubit1]

        H_main_qubits = [
        H_no_time,
        qubit1_qubit2_adj_H,
        qubit1_adj_qubit2_H
        ]

        qubit3_qubit2_adj_H = qs.modes_a[qubit3]*qs.modes_a_dag[qubit2]
        qubit3_adj_qubit2_H = qs.modes_a[qubit2]*qs.modes_a_dag[qubit3]
        qubit4_qubit2_adj_H = qs.modes_a[qubit4]*qs.modes_a_dag[qubit2]
        qubit4_adj_qubit2_H = qs.modes_a[qubit2]*qs.modes_a_dag[qubit4]
        qubit3_qubit1_adj_H = qs.modes_a[qubit3]*qs.modes_a_dag[qubit1]
        qubit3_adj_qubit1_H = qs.modes_a[qubit1]*qs.modes_a_dag[qubit3]
        qubit4_qubit1_adj_H = qs.modes_a[qubit4]*qs.modes_a_dag[qubit1]
        qubit4_adj_qubit1_H = qs.modes_a[qubit1]*qs.modes_a_dag[qubit4]
        qubit3_qubit4_adj_H = qs.modes_a[qubit3]*qs.modes_a_dag[qubit4]
        qubit3_adj_qubit4_H = qs.modes_a[qubit4]*qs.modes_a_dag[qubit3]

        H_added = [
                qubit3_qubit2_adj_H,
                qubit3_adj_qubit2_H,
                qubit4_qubit2_adj_H,
                qubit4_adj_qubit2_H,
                qubit3_qubit1_adj_H,
                qubit3_adj_qubit1_H,
                qubit4_qubit1_adj_H,
                qubit4_adj_qubit1_H,
                qubit3_qubit4_adj_H,
                qubit3_adj_qubit4_H
                ]
        
        # create the added gates in a more generalized manner
        # for i in range(self.num_qubits):

        
        def choose_lambda(choice):
            for _ in range(len(H_added)):
                if(choice == 2):
                    # H_modified.append(6 * (l1**2) * H_added[i])
                    H_modified = [
                        hamiltonian_coeff * qubit3_qubit2_adj_H,
                        hamiltonian_coeff * qubit3_adj_qubit2_H,
                        hamiltonian_coeff * qubit4_qubit2_adj_H,
                        hamiltonian_coeff * qubit4_adj_qubit2_H,
                        hamiltonian_coeff * qubit3_qubit1_adj_H,
                        hamiltonian_coeff * qubit3_adj_qubit1_H,
                        hamiltonian_coeff * qubit4_qubit1_adj_H,
                        hamiltonian_coeff * qubit4_adj_qubit1_H,
                        hamiltonian_coeff * qubit3_qubit4_adj_H,
                        hamiltonian_coeff * qubit3_adj_qubit4_H
                        ]
                    # H_modified= [6 * (l1**2) *qubit3_qubit4_adj_H,
                    # 6 * (l1**2) *qubit3_adj_qubit4_H]
                # elif(choice == 3):
                    # H_modified.append(hamiltonian_coeff * H_added[i])
                elif(choice == 4):
                    # H_modified.append(6 * (l1**4) * H_added[i])
                    H_modified = [
                        hamiltonian_coeff * qubit3_qubit2_adj_H,
                        hamiltonian_coeff * qubit3_adj_qubit2_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit4_qubit2_adj_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit4_adj_qubit2_H,
                        hamiltonian_coeff * qubit3_qubit1_adj_H,
                        hamiltonian_coeff * qubit3_adj_qubit1_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit4_qubit1_adj_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit4_adj_qubit1_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit3_qubit4_adj_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit3_adj_qubit4_H
                        ]
                    # H_modified= [6 * (l1**4) *qubit3_qubit4_adj_H,
                    # 6 * (l1**4) *qubit3_adj_qubit4_H]
                # elif(choice == 5):
                    # H_modified.append(hamiltonian_coeff * H_added[i])
                elif(choice == 6):
                    # H_modified.append(6 * (l1**6) * H_added[i])
                    H_modified = [
                        hamiltonian_coeff * (self._lambda ** 2) * qubit3_qubit2_adj_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit3_adj_qubit2_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit4_qubit2_adj_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit4_adj_qubit2_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit3_qubit1_adj_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit3_adj_qubit1_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit4_qubit1_adj_H,
                        hamiltonian_coeff * (self._lambda ** 2) * qubit4_adj_qubit1_H,
                        hamiltonian_coeff * (self._lambda ** 4) * qubit3_qubit4_adj_H,
                        hamiltonian_coeff * (self._lambda ** 4) * qubit3_adj_qubit4_H
                        ]
                    # H_modified= [6 * (l1**6) *qubit3_qubit4_adj_H,
                    # 6 * (l1**6) *qubit3_adj_qubit4_H]

            return H_modified

        
        choice = 2 # this is a place holder that needs updated
        H_total = []
        H_total.extend(H_main_qubits)
        # call in the function 
        H_mod = choose_lambda(choice=choice)
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
        qubit1_qubit4_adj_val = int_func(w1,w4,wp,T) + int_func_conj_wp(w1,w4,wp,T)
        qubit2_qubit4_adj_val = int_func(w2,w4,wp,T) + int_func_conj_wp(w2,w4,wp,T)
        qubit1_adj_qubit4_val = int_func(w4,w1,wp,T) + int_func_conj_wp(w4,w1,wp,T)
        qubit2_adj_qubit4_val = int_func(w4,w2,wp,T) + int_func_conj_wp(w4,w2,wp,T)
        qubit3_qubit4_adj_val = int_func(w3,w4,wp,T) + int_func_conj_wp(w3,w4,wp,T)
        qubit3_adj_qubit4_val = int_func(w4,w3,wp,T) + int_func_conj_wp(w4,w3,wp,T)

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

        num_module = 0 # this is a placeholder this will be specific to the topology selected (this is the point at which to calibrate the gate)
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
        # self.amp = T * hamiltonian_coeff * (np.pi / 2)
        # self.ampi = 5
        # amps = np.linspace(0, self.ampi, 400)
        # fids = []
        # eta_list = []


        # the issue here is that it is double optimizing which you do not want to do 
        # this means that you have to calibrate the gate first to get what the amp should be and then keep it fixed for all of the frequency tests and this should give a good result 
        # non-optimization cell 
        Z = []
        Z.clear()
        for j in range(len(H)):
            
            Z.append(eta * H[j] * multiplier_times[j])

        w = sum(Z)
        U_propagator = (-1j * w).expm()

        #calculate the fidelity
        fid = np.abs(qt.average_gate_fidelity(desired_U, U_propagator))
        # results.append([i, fid])

        return [fid, eta]

    
    def result_calibration(self):
        return [(self.qubit1_freq, self.qubit2_freq, self.qubit3_freq, self.qubit4_freq, self.max_fid, self.eta_max)]
    
    def min_sep(self):
        # this can be made better once the qubit setup is better known but for now this only works for a four qubit setup
        gate12 = np.abs(self.w1 - self.w2)
        gate13 = np.abs(self.w1 - self.freq_3)
        gate23 = np.abs(self.w2 - self.freq_3)
        gate14 = np.abs(self.w1 - self.freq_4)
        gate24 = np.abs(self.w2 - self.freq_4)
        gate34 = np.abs(self.freq_3 - self.freq_4)

        sep12_13 = np.abs(gate12 - gate13)
        sep12_23 = np.abs(gate12 - gate23)
        sep12_14 = np.abs(gate12 - gate14)
        sep12_24 = np.abs(gate12 - gate24)
        sep12_34 = np.abs(gate12 - gate34)
        sep13_23 = np.abs(gate13 - gate23)
        sep13_14 = np.abs(gate13 - gate14)
        sep13_24 = np.abs(gate13 - gate24)
        sep13_34 = np.abs(gate13 - gate34)
        sep23_14 = np.abs(gate23 - gate14)
        sep23_24 = np.abs(gate23 - gate24)
        sep23_34 = np.abs(gate23 - gate34)
        sep14_24 = np.abs(gate14 - gate24)
        sep14_34 = np.abs(gate14 - gate34)
        sep24_34 = np.abs(gate24 - gate34)

        # seps = [sep12_13, sep12_23, sep13_23, sep12_14, sep12_34, sep12_24, sep13_14, sep13_24, sep13_34, sep23_14, sep23_24, sep23_34, sep14_24, sep14_34, sep24_34]
        seps = [sep12_34, sep13_34, sep23_34, sep14_34, sep24_34]
        return min(seps)
