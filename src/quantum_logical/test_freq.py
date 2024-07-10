from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from quantum_logical import Pulse, DressedQuantumSystem
from quantum_logical.hamiltonian import QubitSNAILModule
from quantum_logical.mode import QubitMode, SNAILMode
from qutip import Options
from multiprocessing.pool import ThreadPool

from tqdm import tqdm
from scipy.optimize import minimize
import scienceplots


class Test_freq:
    def __init__(self, freqs, store):
        self.store = store
        qubit_dim = 2
        qubit1 = QubitMode(name="q1", dim=qubit_dim, freq=freqs[0])
        qubit2 = QubitMode(name="q2", dim=qubit_dim, freq=freqs[1])
        qubit3 = QubitMode(name="q3", dim=qubit_dim, freq=freqs[2])
        self.qubits = [qubit1, qubit2, qubit3]
        self.snail = SNAILMode(name="s", freq=freqs[3], g3=0.3, dim=10, T1=1e3, T2=5e2)

        # define couplings so hybridizations are all equal
        # g/delta = 0.1 for all qubits
        g2_0 = 0.1 * np.abs(self.snail.freq - qubit1.freq)
        g2_1 = 0.1 * np.abs(self.snail.freq - qubit2.freq)
        g2_2 = 0.1 * np.abs(self.snail.freq - qubit3.freq)
        _couplings = {
            frozenset([qubit1, self.snail]): g2_0,
            frozenset([qubit2, self.snail]): g2_1,
            frozenset([qubit3, self.snail]): g2_2,
        }

        self.qs = DressedQuantumSystem(
        self.qubits + [self.snail], couplings=_couplings, hamiltonian_cls=QubitSNAILModule)


        # (undressed) expectation operators
        e_ops = [self.qs.modes_num[m] for m in self.qs.modes]

        # collapse operators
        self.c_ops = []
        # for mode in qs.modes:
        #    c_ops.append(mode.collapse_operators(qs))

        # create an initial state
        mode_states = tuple([(qubit1, 0), (qubit2, 1), (qubit3, 0)])
        psi0_a = self.qs.prepare_approx_state(mode_states)
        # mode_states = tuple([(qubit1, 0), (qubit2, 0), (qubit3, 1)])
        # psi0_b = qs.prepare_approx_state(mode_states)
        # psi0 = (psi0_a + psi0_b).unit()
        self.psi0 = psi0_a.unit()
        rho0 = self.psi0 * self.psi0.dag()

        from qutip_qip.operations import iswap

        desired_U = iswap()  # The iSWAP gate for a 2-qubit system

        # Create isometries for qubit 1 and qubit 2 to extend the {g, e} subspace action to the full qubit space
        identity_isometry = (
            qt.basis(qubit_dim, 0) * qt.basis(2, 0).dag()
            + qt.basis(qubit_dim, 1) * qt.basis(2, 1).dag()
        )
        identity_isometry = qt.tensor(identity_isometry, identity_isometry)

        # Apply the isometry to extend the gate action to the complete system
        extended_q1_q2 = identity_isometry * desired_U * identity_isometry.dag()

        # Tensor with identity matrices for the remaining qubits and the SNAIL mode
        # Skip the first two qubits as they're already included
        # Skip the last index as it's the SNAIL mode
        for mode in self.qs.modes[2:-1]:
            extended_q1_q2 = qt.tensor(extended_q1_q2, qt.qeye(mode.dim))

        # The extended_iswap_q1_q2 now acts as the desired iSWAP gate on {g, e} of qubits 1 and 2, and as identity on the rest
        desired_U = extended_q1_q2

        # act on qubit space only
        qubit_rho0 = rho0.ptrace(range(len(self.qubits)))
        self.expected_qubit_rho = qt.Qobj(desired_U * qubit_rho0 * desired_U.dag())

        self.width_d = 300  # ns
        self.off_d = 20
        # args = {"shape": Pulse.box, "shape_params": {"t0": off_d, "width": width_d}}
        args = {"shape": Pulse.smoothbox, "shape_params": {"t0": self.off_d, "width": self.width_d}}
        self.full_time = np.linspace(0, self.width_d + 2 * self.off_d, 500)
        self.wp = np.abs(qubit1.freq - qubit2.freq)
        pulse = Pulse(omega=self.wp, amp=6.5)

        # Plot the Gaussian pulse shape
        # Pulse.plot_pulse([(pulse, args)], full_time)

        # single period Pulse
        self.wp = np.abs(qubit1.freq - qubit2.freq)
        T = 2 * np.pi / self.wp
        period_time = np.linspace(0, T, 250)  # a single period of the pulse
        args = {"shape": Pulse.constant}
        pulse = Pulse(omega=self.wp, amp=6.5)
        # Pulse.plot_pulse([(pulse, args)], period_time)
        def _construct_propagator(omega_amp_tuple):
            omega, amp = omega_amp_tuple
            T = 2 * np.pi / omega
            pulse = Pulse(omega=omega, amp=amp)
            H_pump = self.qs.hamiltonian.driven_term(snail_mode=self.snail)
            H = [self.qs.hamiltonian.H0, [H_pump, pulse.drive]]
            U_t = qt.propagator(H, T, self.c_ops, args=args)

            # off pump
            pulse_off = Pulse(omega=omega, amp=0)
            H_off = [self.qs.hamiltonian.H0, [H_pump, pulse_off.drive]]
            U_off = qt.propagator(H_off, self.off_d, self.c_ops, args=args)
            return U_t, U_off


        def _construct_full_propagator(omega_amp_tuple):
            # create total time evolution operator
            omega, amp = omega_amp_tuple
            T = 2 * np.pi / omega
            U_t, U_off = _construct_propagator(omega_amp_tuple)
            n_periods = int(self.width_d / T)
            U_total = U_off * (U_t**n_periods) * U_off
            return U_total, self.full_time[-1]


        def propagator_task(omega_amp_tuple) -> qt.Qobj:
            omega, amp = omega_amp_tuple
            T = 2 * np.pi / omega
            n_periods = int(self.width_d / T)
            U_t, U_off = _construct_propagator((omega, amp))
            states = [rho0]
            times = [0]

            # initial off pulse
            rho_tt = qt.Qobj(U_off * rho0 * U_off.dag())
            states.append(rho_tt)
            times.append(self.off_d)

            for _ in range(n_periods):
                rho_tt = qt.Qobj(U_t * rho_tt * U_t.dag())
                states.append(rho_tt)
                times.append(times[-1] + T)

            # final off pulse
            rho_tt = qt.Qobj(U_off * rho_tt * U_off.dag())
            states.append(rho_tt)
            times.append(times[-1] + self.off_d)

            return [(t, state) for t, state in zip(times, states)]


        def plot_expected_occupations(result_obj):
            # Prepare plot
            plt.figure(figsize=(6, 6))

            times = [t for t, _ in result_obj]

            # Collect data for all modes
            for mode in self.qs.modes_num:
                populations = [
                    np.abs(qt.expect(self.qs.modes_num[mode], state)) for _, state in result_obj
                ]
                # Plot each mode's population over time
                plt.plot(times, populations, label=mode.name, marker="o")

            # Setting plot labels and title
            plt.xlabel("Time (ns)")
            plt.ylabel("Expected Population")
            plt.title("Expected Population of Modes Over Time")
            plt.legend()
            plt.grid(True)

            # Show plot
            plt.show()

            # create the optimizer and then run the result of the optimizer in the bar below
        detuning = np.linspace(-.07, .07, 30)
        amp = np.linspace(8, 16, 30)
        amp_detuning = [(i, j) for i in detuning for j in amp]
        self.fids = []
        self.fids.clear()

        results = qt.parallel.parallel_map(self.sim_task, amp_detuning, progress_bar='enhanced')

        max = 0.0
        for i in range(len(results)):
            fid = results[i][2]
            if fid > max:
                amplitude = results[i][1]
                detune = results[i][0]

        # detuning = (25/3) * 2 * np.pi / 1000
        # amplitude = (125/9)
        me_result = self.mesolve_task((self.wp + detune, amplitude))
        # plot_expected_occupations(me_result)
        result = self.extract_state_fidelity(me_result)
        self.result(result=result)
        return None
        # this will need to be optimized because right now it does not make sense that this can be done over it

    def mesolve_task(self, omega_amp_tuple):
        omega, amp = omega_amp_tuple
        pulse = Pulse(omega=omega, amp=amp)
        args = {"shape": Pulse.smoothbox, "shape_params": {"t0": self.off_d, "width": self.width_d}}
        H_pump = self.qs.hamiltonian.driven_term(snail_mode=self.snail)
        H = [self.qs.hamiltonian.H0, [H_pump, pulse.drive]]
        result = qt.mesolve(H, self.psi0, self.full_time, self.c_ops, args=args)
        return [(t, state) for t, state in zip(result.times, result.states)]

    # final state fidelity
    def extract_state_fidelity(self, result_obj):
        final_state = result_obj[-1][1]
        qubit_rhof = final_state.ptrace(range(len(self.qubits)))
        return qt.fidelity(qubit_rhof,self.expected_qubit_rho)
        

    def sim_task(self, amp_detuning):
        detuning, amp = amp_detuning

        me_result = self.mesolve_task((self.wp + detuning, amp))
        fid = self.extract_state_fidelity(me_result)

        self.fids.append(fid)
        return [detuning, amp, fid]
    
    def result(self, result):
        print(result)
        self.store.append(result)
        return None

            

        # # amplitude = results.index(results.max(axis = 0)[2])[1]
        # # detuned = results.index(results.max(axis = 0)[2])[0]

        # res = []
        # for i in tqdm(range(len(amp_detuning))):
        #     detuning, amp = amp_detuning[i]

        #     me_result = mesolve_task((wp + detuning, amp))
        #     fid = extract_state_fidelity(me_result)

        #     fids.append(fid)

        #     res.append([detuning, amp, fid])

        # data analysis from above