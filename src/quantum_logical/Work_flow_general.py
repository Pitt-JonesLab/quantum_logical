import numpy as np
import qutip as qt
from quantum_logical.pulsesim import QuantumSystem
from quantum_logical.pulsesim.mode import QubitMode, SNAILMode
from itertools import product
from qutip.qip.operations import iswap


class Work_flow1():
    def __init__(self, num_qubits, lamb, pumped_qubits):
        self.pumped_qubits = pumped_qubits
        self.num_qubits = num_qubits
        self._lambda = lamb

        return None
    
    def calibrator(self, freq_list):
        # generalize these next few lines
        self.freq_list = freq_list
        dim = 2
        qubits = []

        if self.num_qubits == len(self.freq_list):
            for i in range(self.num_qubits):
                qubits.append(QubitMode(mode_type="Qubit",
                                        name=f"q{i}", dim=dim, freq=self.freq_list[i], alpha=-0.161, T1=1e2, T2=5e1))
        else:
            raise Exception("Number of qubits does not match the amount of frequencies given")

        ws = self.freq_list[0] + ((self.freq_list[0] - self.freq_list[1]) / 3)
        # ws = 6 - (1 / 3) * (1 / 2)

        # coupling_list = [2 * np.pi * 0.05467, 2 * np.pi * 0.0435, 2 * np.pi * 0.04875, 2 * np.pi * 0.04875, 2 * np.pi * 0.04875]

        snail = SNAILMode(mode_type="Snail", name="s", freq=ws, g3=0.3, dim=10, T1=1e3, T2=5e2)
        _couplings = {
            # this needs to be generalized but this involves a bit more thought
            frozenset([qubits[0], snail]): 2 * np.pi * 0.05467,
            frozenset([qubits[1], snail]): 2 * np.pi * 0.0435,
            frozenset([qubits[2], snail]): 2 * np.pi * 0.04875,
            frozenset([qubits[3], snail]): 2 * np.pi * 0.04875

        }

        qs = QuantumSystem(qubits + [snail], couplings=_couplings)
        # time over which to optimize
        T = 100

        g_3 = 0.3  # this should be given above 

        # pump(drive) frequency
        # this should be made more specific to different qubits 
        # this statement above may introduce a complexity that is not necessary 
        # wp = w1 - w2
        wp = np.abs(self.freq_list[self.pumped_qubits[0] - 1] - self.freq_list[self.pumped_qubits[1] - 1]) 

        hamiltonian_coeff = 6 * (self._lambda ** 2) * g_3 # check this line against the crowding code because they should spit out the same thing and they do not 


        # Creating the main hamiltonian terms with the driven gate
        H_no_time = hamiltonian_coeff * (qs.modes_a[qubits[self.pumped_qubits[0] - 1]]*qs.modes_a_dag[qubits[self.pumped_qubits[1] - 1]] 
                                         + qs.modes_a[qubits[self.pumped_qubits[1] - 1]]*qs.modes_a_dag[qubits[self.pumped_qubits[0] - 1]])

        #terms that come from the gate that is desired that do not go to one
        qubit1_qubit2_adj_H =(hamiltonian_coeff 
                              * qs.modes_a[qubits[self.pumped_qubits[0] - 1]]
                              * qs.modes_a_dag[qubits[self.pumped_qubits[1] - 1]])
        qubit1_adj_qubit2_H = (hamiltonian_coeff 
                               * qs.modes_a[qubits[self.pumped_qubits[1] - 1]]
                               * qs.modes_a_dag[qubits[self.pumped_qubits[0] - 1]])

        H_main_qubits = [
            H_no_time,
            qubit1_qubit2_adj_H,
            qubit1_adj_qubit2_H
            ]
 
        H_total = []
        H_total.extend(H_main_qubits)
        len(H_total)

        # build the pulses for the unitary 
        def int_func(w1, w2, wp, t):
            a=((np.exp(-1j*(w1-w2+wp)*t))/(-1j*(w1-w2+wp)) - (1/(-1j*(w1-w2+wp))))
            return a

        def int_func_conj_wp(w1, w2, wp, t):
            a = ((np.exp(-1j*(w1-w2-wp)*t))/(-1j*(w1-w2-wp)) - (1/(-1j*(w1-w2-wp))))
            return a

        def int_func_conj(w1, w2, wp, t):
            a = ((np.exp(1j*(w1-w2+wp)*t))/(1j*(w1-w2+wp)) - (1/(1j*(w1-w2+wp))))
            return a 

        def int_func_conj_wp_conj(w1, w2, wp, t):
            a = ((np.exp(1j*(w1-w2-wp)*t))/(1j*(w1-w2-wp)) - (1/(1j*(w1-w2-wp))))
            return a

        T = 100

        qubit1_qubit2_adj_val = int_func_conj_wp(self.freq_list[0],self.freq_list[1],wp,T)
        qubit1_adj_qubit2_val = int_func_conj_wp_conj(self.freq_list[0],self.freq_list[1],wp,T)

        # building the time_multiplier list 
        T_mult = [
            T,
            qubit1_qubit2_adj_val,
            qubit1_adj_qubit2_val]

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

        # Tensor with identity matrices for the remaining modes
        for mode in qs.modes[2:]:  # Skip the first two qubits(included)
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


        # optimizing to find proper gate calibration parameters
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
    
    # starting the actual experimental function
    def work_experiment1(self, eta, lambda_pow, spectator_gate_count, freq_values):
        #  this function needs generalized before it can be used in a good way
        # generalize these next few lines
        dim = 2
        qubits = []

        if self.num_qubits == len(freq_values):
            for i in range(self.num_qubits):
                qubits.append(QubitMode(mode_type="Qubit",
                                        name=f"q{i}", dim=dim, freq=freq_values[i], alpha=-0.161, T1=1e2, T2=5e1))
        else:
            raise Exception("Number of qubits does not match the amount of frequencies given")

        ws = freq_values[0] + ((freq_values[0] - freq_values[1]) / 3)

        snail = SNAILMode(mode_type="Snail", name="s", freq=ws, g3=0.3, dim=10, T1=1e3, T2=5e2)
        _couplings = {None}  # keep in mind that this was not None incase there is something wrong 

        qs = QuantumSystem(qubits + [snail], couplings=_couplings)
        # time over which to optimize
        T = 100

        g_3 = .3  # this should be given above 

        # pump(drive) frequency
        wp = np.abs(freq_values[self.pumped_qubits[0] - 1] - freq_values[self.pumped_qubits[1] - 1])

        hamiltonian_coeff = 6 * (self._lambda ** 2) * g_3

        # Creating the main hamiltonian terms with the driven gate
        H_no_time = hamiltonian_coeff * (qs.modes_a[qubits[self.pumped_qubits[0] - 1]]*qs.modes_a_dag[qubits[self.pumped_qubits[1] - 1]] 
                                         + qs.modes_a[qubits[self.pumped_qubits[1] - 1]]*qs.modes_a_dag[qubits[self.pumped_qubits[0] - 1]])

        # terms that come from the gate that is desired that do not go to one
        qubit1_qubit2_adj_H = (hamiltonian_coeff 
                               * qs.modes_a[qubits[self.pumped_qubits[0] - 1]]
                               * qs.modes_a_dag[qubits[self.pumped_qubits[1] - 1]])
        qubit1_adj_qubit2_H = (hamiltonian_coeff 
                               * qs.modes_a[qubits[self.pumped_qubits[1] - 1]]
                               * qs.modes_a_dag[qubits[self.pumped_qubits[0] - 1]])

        H_main_qubits = [
            H_no_time,
            qubit1_qubit2_adj_H,
            qubit1_adj_qubit2_H
            ]
        
        # this is the part of the code that needs to generalized because this far everything else has been generalized

        # the important qubits are the last two in the list those are the ones that are moving around

        def choose_distance(power):
            H_added_gates = []
            for i in range(self.num_qubits):
                for j in range(self.num_qubits):
                    if i == j or (i == (self.pumped_qubits[0] - 1) and j == (self.pumped_qubits[1] - 1)) or (i == (self.pumped_qubits[1] - 1) and j == (self.pumped_qubits[0] - 1)):
                        # do nothing
                        pass
                    else:
                        if power == 2:
                            H_added_gates.append((hamiltonian_coeff * (qs.modes_a[qubits[i]] * qs.modes_a_dag[qubits[j]])))
                        elif power == 4:
                            if (i == (len(freq_values) - 1) or j == (len(freq_values) - 1)) and (i != (len(freq_values) - 1) or j != (len(freq_values) - 1)):
                                H_added_gates.append((hamiltonian_coeff * (self._lambda ** 2) * (qs.modes_a[qubits[i]] * qs.modes_a_dag[qubits[j]])))
                            else:
                                H_added_gates.append((hamiltonian_coeff * (qs.modes_a[qubits[i]] * qs.modes_a_dag[qubits[j]])))
                        elif power == 6:
                            H_added_gates.append((hamiltonian_coeff * (self._lambda ** 4) * (qs.modes_a[qubits[i]] * qs.modes_a_dag[qubits[j]])))

            return H_added_gates

        pow = lambda_pow # this is a place holder that needs updated
        H_total = []
        H_total.extend(H_main_qubits)
        # call in the function 
        H_mod = choose_distance(power=pow)
        H_total.extend(H_mod)


        # everything that was done above to build the gates should be done to build the pulses since they also have a set pattern
        # build the pulses for the unitary 
        def int_func(w1, w2, wp, t):
            a=((np.exp(-1j*(w1-w2+wp)*t))/(-1j*(w1-w2+wp)) - (1/(-1j*(w1-w2+wp))))
            return a

        def int_func_conj_wp(w1, w2, wp, t):
            a = ((np.exp(-1j*(w1-w2-wp)*t))/(-1j*(w1-w2-wp)) - (1/(-1j*(w1-w2-wp))))
            return a

        def int_func_conj(w1, w2, wp, t):
            a = ((np.exp(1j*(w1-w2+wp)*t))/(1j*(w1-w2+wp)) - (1/(1j*(w1-w2+wp))))
            return a 

        def int_func_conj_wp_conj(w1, w2, wp, t):
            a = ((np.exp(1j*(w1-w2-wp)*t))/(1j*(w1-w2-wp)) - (1/(1j*(w1-w2-wp))))
            return a

        T = 100

        T_list = []
        T_list.append(T)
        T_list.append(int_func_conj_wp(freq_values[self.pumped_qubits[0] - 1], freq_values[self.pumped_qubits[1] - 1], wp, T))
        T_list.append(int_func_conj_wp_conj(freq_values[self.pumped_qubits[0] - 1], freq_values[self.pumped_qubits[1] - 1], wp, T))
        for i in range(len(qubits)):
            for j in range(len(qubits)):
                if i == j or (i == (self.pumped_qubits[0] - 1) and j == (self.pumped_qubits[1] - 1)) or (i == (self.pumped_qubits[1] - 1) and j == (self.pumped_qubits[0] - 1)):
                        # do nothing
                    pass
                else:
                    T_list.append(int_func(freq_values[i], freq_values[j], wp, T) + int_func_conj_wp(freq_values[i], freq_values[j], wp, T))

        def mod_amount(count):
            H_tot = []
            ts = []
            ts.append(T_list[0])
            ts.append(T_list[1])
            ts.append(T_list[2])
            H_tot.extend(H_main_qubits)
            for _ in range(count):
                for i in range(3, len(H_total)):
                    H_tot.append(H_total[i])
                    ts.append(T_list[i])

            return [H_tot, ts]

        num_module = spectator_gate_count # this is a placeholder this will be specific to the topology selected (this is the point at which to calibrate the gate)
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

        # run the same experiment as above but this time do it for the qubits that have been passed in and see if you can get a better result 


        return [freq_values, fid]
    
    # this is the alt iswap function that is meant to decide if the qubit 3 and 4 pairing above is actually viable or does it only work for the one gate that is done above
    
    def alt_iswap(self, lambda_pow, spectator_gate_count, freq_values, amp_max):

        dim = 2
        qubits = []

        if self.num_qubits == len(freq_values):
            for i in range(self.num_qubits):
                qubits.append(QubitMode(mode_type="Qubit",
                                        name=f"q{i}", dim=dim, freq=freq_values[i], alpha=-0.161, T1=1e2, T2=5e1))
        else:
            raise Exception("Number of qubits does not match the amount of frequencies given")

        ws = freq_values[0] + ((freq_values[0] - freq_values[1]) / 3)

        snail = SNAILMode(mode_type="Snail", name="s", freq=ws, g3=0.3, dim=10, T1=1e3, T2=5e2)
        _couplings = {None}  # keep in mind that this was not None incase there is something wrong 

        qs = QuantumSystem(qubits + [snail], couplings=_couplings)
        # time over which to optimize
        T = 100

        g_3 = .3  # this should be given above 

        # pump(drive) frequency
        wp = freq_values[self.pumped_qubits[1] - 1] - freq_values[self.pumped_qubits[0] - 1]

        hamiltonian_coeff = 6 * (self._lambda ** 2) * g_3

        # Creating the main hamiltonian terms with the driven gate
        H_no_time = hamiltonian_coeff * (qs.modes_a[qubits[self.pumped_qubits[0] - 1]]*qs.modes_a_dag[qubits[self.pumped_qubits[1] - 1]] 
                                         + qs.modes_a[qubits[self.pumped_qubits[1] - 1]]*qs.modes_a_dag[qubits[self.pumped_qubits[0] - 1]])

        # terms that come from the gate that is desired that do not go to one
        qubit1_qubit2_adj_H = (hamiltonian_coeff 
                               * qs.modes_a[qubits[self.pumped_qubits[0] - 1]]
                               * qs.modes_a_dag[qubits[self.pumped_qubits[1] - 1]])
        qubit1_adj_qubit2_H = (hamiltonian_coeff 
                               * qs.modes_a[qubits[self.pumped_qubits[1] - 1]]
                               * qs.modes_a_dag[qubits[self.pumped_qubits[0] - 1]])

        H_main_qubits = [
            H_no_time,
            qubit1_qubit2_adj_H,
            qubit1_adj_qubit2_H
            ]
        
        # this is the part of the code that needs to generalized because this far everything else has been generalized

        # the important qubits are the last two in the list those are the ones that are moving around

        def choose_distance(power):
            H_added_gates = []
            for i in range(self.num_qubits):
                for j in range(self.num_qubits):
                    if i == j or (i == (self.pumped_qubits[0] - 1) and j == (self.pumped_qubits[1] - 1)) or (i == (self.pumped_qubits[1] - 1) and j == (self.pumped_qubits[0] - 1)):
                        # do nothing
                        pass
                    else:
                        if power == 2:
                            H_added_gates.append((hamiltonian_coeff * (qs.modes_a[qubits[i]] * qs.modes_a_dag[qubits[j]])))
                        elif power == 4:
                            if (i == (len(freq_values) - 1) or j == (len(freq_values) - 1)) and (i != (len(freq_values) - 1) or j != (len(freq_values) - 1)):
                                H_added_gates.append((hamiltonian_coeff * (self._lambda ** 2) * (qs.modes_a[qubits[i]] * qs.modes_a_dag[qubits[j]])))
                            else:
                                H_added_gates.append((hamiltonian_coeff * (qs.modes_a[qubits[i]] * qs.modes_a_dag[qubits[j]])))
                        elif power == 6:
                            H_added_gates.append((hamiltonian_coeff * (self._lambda ** 4) * (qs.modes_a[qubits[i]] * qs.modes_a_dag[qubits[j]])))

            return H_added_gates

        pow = lambda_pow # this is a place holder that needs updated
        H_total = []
        H_total.extend(H_main_qubits)
        # call in the function 
        H_mod = choose_distance(power=pow)
        H_total.extend(H_mod)


        # everything that was done above to build the gates should be done to build the pulses since they also have a set pattern
        # build the pulses for the unitary 
        def int_func(w1, w2, wp, t):
            a=((np.exp(-1j*(w1-w2+wp)*t))/(-1j*(w1-w2+wp)) - (1/(-1j*(w1-w2+wp))))
            return a

        def int_func_conj_wp(w1, w2, wp, t):
            a = ((np.exp(-1j*(w1-w2-wp)*t))/(-1j*(w1-w2-wp)) - (1/(-1j*(w1-w2-wp))))
            return a

        def int_func_conj(w1, w2, wp, t):
            a = ((np.exp(1j*(w1-w2+wp)*t))/(1j*(w1-w2+wp)) - (1/(1j*(w1-w2+wp))))
            return a 

        def int_func_conj_wp_conj(w1, w2, wp, t):
            a = ((np.exp(1j*(w1-w2-wp)*t))/(1j*(w1-w2-wp)) - (1/(1j*(w1-w2-wp))))
            return a

        T = 100

        T_list = []
        T_list.append(T)
        T_list.append(int_func_conj_wp(freq_values[self.pumped_qubits[0] - 1], freq_values[self.pumped_qubits[1] - 1], wp, T))
        T_list.append(int_func_conj_wp_conj(freq_values[self.pumped_qubits[0] - 1], freq_values[self.pumped_qubits[1] - 1], wp, T))
        for i in range(len(qubits)):
            for j in range(len(qubits)):
                if i == j or (i == (self.pumped_qubits[0] - 1) and j == (self.pumped_qubits[1] - 1)) or (i == (self.pumped_qubits[1] - 1) and j == (self.pumped_qubits[0] - 1)):
                        # do nothing
                    pass
                else:
                    T_list.append(int_func(freq_values[i], freq_values[j], wp, T) + int_func_conj_wp(freq_values[i], freq_values[j], wp, T))

        def mod_amount(count):
            H_tot = []
            ts = []
            ts.append(T_list[0])
            ts.append(T_list[1])
            ts.append(T_list[2])
            H_tot.extend(H_main_qubits)
            for _ in range(count):
                for i in range(3, len(H_total)):
                    H_tot.append(H_total[i])
                    ts.append(T_list[i])

            return [H_tot, ts]

        num_module = spectator_gate_count # this is a placeholder this will be specific to the topology selected (this is the point at which to calibrate the gate)
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


        # run the fidelity analysis over the expected gate 
        # self.amp = T * hamiltonian_coeff * (np.pi / 2)
        ampi = amp_max
        amps = np.linspace(0, ampi, 300)
        fids = []
        eta_list = []


        # optimizing to find proper gate calibration parameters
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
        # results.append([i, fid])

        return max_fid

