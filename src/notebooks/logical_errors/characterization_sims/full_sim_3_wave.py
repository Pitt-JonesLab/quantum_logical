import qutip as qt
from qutip import tensor, basis, qeye
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit
from itertools import product
from quantum_logical.gate_extender import Gate_extender, Convert_levels
from quantum_logical.trotterization import Trotterization


def gate( dim, N):
    # creating the gates
    dim = dim
    N = N
    # creating the set of cnots (this will operate between the g and e levels)
    cnot_create = Convert_levels(num_qubits=N)
    cnot1 = cnot_create.Cnot(dim=3, target=1, control=0, high=2, low=0)
    cnot2 = cnot_create.Cnot(dim=3, target=2, control=0, high=2, low=0)

    cnot3 = cnot_create.Cnot(dim=3, target=3, control=0, high=1, low=0)
    cnot4 = cnot_create.Cnot(dim=3, target=3, control=1, high=1, low=0)
    

    cnot5 = cnot_create.Cnot(dim=3, target=4, control=1, high=1, low=0)
    cnot6 = cnot_create.Cnot(dim=3, target=4, control=2, high=1, low=0)

    # the x_gate needs to be made in a qutrit gate and will involve conversion 
    x_gate = qt.Qobj([[0, 1],[1, 0]])



    hada = qt.Qobj([[1/np.sqrt(2), 0, 1/np.sqrt(2)], [0, 1, 0], [1/np.sqrt(2), 0, -1/np.sqrt(2)]])



    gate_extention = Gate_extender(num_qubits=1)
    x_gate = gate_extention.qubit_to_qudit(gate=x_gate, from_dim=2, to_dim=dim)


    # conversion of some of the gates into the qutrit space 
    new_dim = dim
    converter = Convert_levels(num_qubits=1)
    x_gate = converter.level_conversion(levels=[0,2], dim=dim, gate=x_gate, qubits=None)


    x_layer = tensor(tensor([x_gate] * 3), tensor([qeye(new_dim)] * 2))
    hada_layer = tensor(tensor([hada] * 3), tensor([qeye(new_dim)] * 2))

    # building the correction z_gate 
    correction_x = qt.Qobj([[0, 1],[1, 0]])
    gate_extention = Gate_extender(num_qubits=1)
    correction_x = gate_extention.qubit_to_qudit(gate=correction_x, from_dim=2, to_dim=3)
    converter = Convert_levels(num_qubits=1)
    correction_x = converter.level_conversion(levels=[1,2], dim=new_dim, gate=correction_x, qubits=None)

    correction_z = (hada * correction_x * hada.dag())

    # vector setup 
    basis0 = qt.Qobj([[1],[0],[0]])
    basis1 = qt.Qobj([[0],[1],[0]])
    basis2 = qt.Qobj([[0],[0],[1]])
    vector0 = hada * basis0
    vector1 = hada * basis1
    vector2 = hada * basis2
    vectors = [vector0, vector1, vector2]
    vectors = [tensor(i,j,k) for i in vectors for j in vectors for k in vectors]

    cnots = [cnot1, cnot2, cnot3, cnot4, cnot5, cnot6]
    return cnots, correction_z, hada_layer, x_layer, vectors


def state(alpha, beta, dim, hada_layer, cnots, state_choice):
    # creating initial state
    # state controls 
    cnot1 = cnots[0]
    cnot2 = cnots[1]
    if state_choice == "220":
        # state
        psi0 = tensor((alpha * basis(dim, 0) + beta * basis(dim, 2)).unit(), tensor([basis(dim, 0)] * (N-1)))
        rho0 = psi0 * psi0.dag()
        encoded_rho = rho_encoded = hada_layer * cnot1 * rho0 * cnot1.dag() * hada_layer.dag()
        ref_state = hada_layer * cnot2 * cnot1 * rho0 * cnot1.dag() * cnot2.dag() * hada_layer.dag()
        rho_encoded_traced = qt.ptrace(rho_encoded, [0,1,2])

    elif state_choice == "202":
        # state
        psi0 = tensor((alpha * basis(dim, 0) + beta * basis(dim, 2)).unit(), tensor([basis(dim, 0)] * (N-1)))
        rho0 = psi0 * psi0.dag()
        encoded_rho = rho_encoded = hada_layer * cnot2 * rho0 * cnot2.dag() * hada_layer.dag()
        ref_state = hada_layer * cnot2 * cnot1 * rho0 * cnot1.dag() * cnot2.dag() * hada_layer.dag()
        rho_encoded_traced = qt.ptrace(rho_encoded, [0,1,2])
    elif state_choice == "222":

        # state
        psi0 = tensor((alpha * basis(dim, 0) + beta * basis(dim, 2)).unit(), tensor([basis(dim, 0)] * (N-1)))
        rho0 = psi0 * psi0.dag()
        encoded_rho = rho_encoded = hada_layer * cnot2 * cnot1 * rho0 * cnot1.dag() * cnot2.dag() * hada_layer.dag()
        ref_state = hada_layer * cnot2 * cnot1 * rho0 * cnot1.dag() * cnot2.dag() * hada_layer.dag()
        rho_encoded_traced = qt.ptrace(rho_encoded, [0,1,2])


    return rho_encoded, ref_state


def sim_func(values):

    cnots = values[0]
    rho_encoded = values[1]
    ref_state = values[2]
    arrays = values[3]
    dim = values[4]
    N = values[5]
    x_layer = values[6]
    correction_z = values[7]
    hada_layer = values[8]
    vectors = values[9]

    trotter_dt = .005
    cnot3 = cnots[0]
    cnot4 = cnots[1]
    cnot5 = cnots[2]
    cnot6 = cnots[3]

    # initializing Trotterization
    trotter = Trotterization(trotter_dt=trotter_dt, T1=arrays[0], T2=arrays[1], dim=dim, num_qubits=N, qudit="qutrit")
    
    # stored information
    states = []
    no_error_states = []

    # gate setups 
    gates = [[x_layer], [cnot3], [cnot4], [cnot5], [cnot6], [x_layer]]
    cnot_time = .5
    gate_times = [.025, cnot_time, cnot_time, cnot_time, cnot_time, .025]
    # gates = [[x_layer], [cnot3, cnot5], [cnot4, cnot6], [x_layer]]
    # cnot_time = .5
    # gate_times = [.025, cnot_time, cnot_time, .025]

    rho_enc = rho_encoded
    # stabilizer extraction
    rho_encoded = hada_layer * rho_encoded * hada_layer.dag()
    rho_encoded1 = hada_layer * rho_enc * hada_layer.dag() # point of reference when calculating fidelity 

    for i in range(len(gates)):
        rho_evo = trotter.apply(rho=rho_encoded, duration=gate_times[i], unitary=gates[i], errors=True)
        rho_evo_no_error = trotter.apply(rho=rho_encoded1, duration=gate_times[i], unitary=gates[i], errors=False)
        states.extend(rho_evo)
        no_error_states.extend(rho_evo_no_error)
        rho_encoded = rho_evo[-1]
        rho_encoded1 = rho_evo_no_error[-1]
        
    states.append(hada_layer * rho_encoded * hada_layer.dag())
    no_error_states.append(hada_layer * rho_encoded1 * hada_layer.dag())


    # projection operators  
    proj = [qt.tensor(qt.qeye(dim), qt.qeye(dim), qt.qeye(dim), (qt.tensor(qt.basis(dim, i), qt.basis(dim, j)) * (qt.tensor(qt.basis(dim, i), qt.basis(dim, j))).dag())) for i in range(2) for j in range(2)]


    # look into how to get the projection results before moving forward be satisified with it 
    projection_results = [(states[-1] * proj[i]).tr() for i in range(len(proj))]

    # correction_operators
    r00 = qt.tensor([qt.qeye(dim)] * 5)
    r01 = qt.tensor(qt.qeye(dim), qt.qeye(dim), correction_z, qt.tensor([qt.qeye(dim)] * 2))
    r10 = qt.tensor(correction_z, qt.tensor([qt.qeye(dim)] * 4))
    r11 = qt.tensor(qt.qeye(dim), correction_z, qt.qeye(dim), qt.tensor([qt.qeye(dim)] * 2))
    recovery_ops = [[r00], [r01], [r10], [r11]]

    rec = [[proj[0], r00], [proj[1], r01], [proj[2], r10], [proj[3], r11]]

    logicals = [vectors[24], vectors[20], vectors[8], vectors[-1]]
    vals = []
    for vec in logicals:
        val = (vec.dag() * qt.ptrace(states[-1], [0,1,2]) * vec)[0][0][0]
        vals.append(val)
    uncorrectable_error = 1 - np.abs(sum(vals))


    detectable_error = sum([projection_results[i] for i in [0,2,3]]) # how much error is in the system based on what the ancillas say
    
    # imperfect measurement 
    projected_states = []
    measurement_duration = .005
    for i in range(len(proj)):
        projected_state = trotter.apply(states[-1], duration=measurement_duration, unitary=[proj[i]], errors=True)
        projected_states.append(projected_state)

    def neilson_fid(rho, sigma):
        return (((rho.sqrtm()) * sigma * (rho.sqrtm())).sqrtm()).tr()
    
    fid1 = np.abs(neilson_fid(rho=qt.ptrace(states[-1], [0,1,2]), sigma=qt.ptrace(ref_state, [0,1,2])))

    # recovery 
    corrected_states = []
    recovery_duration = .025
    for i in range(len(recovery_ops)):
        corrected_state = trotter.apply(projected_states[i][-1], duration=recovery_duration, unitary=recovery_ops[i], errors=True)
        corrected_states.append(corrected_state)

    for i in range(len(corrected_state)):
        new_state = sum([projection_results[j] * corrected_states[j][i] for j in range(len(projection_results))])
        states.append(new_state)
        no_error_states.append(no_error_states[-1])

    # logicals = [vectors[-1]]
    # vals = []
    # for vec in logicals:
    #     val = (vec.dag() * qt.ptrace(states[-1], [0,1,2]) * vec)[0][0][0]
    #     vals.append(val)
    # errors_after_correction1 = 1 - np.abs(sum(vals))

    # # this is for the errors assuming that the erasure detection and correction works 
    # logicals = [vectors[-1], vectors[25]]
    # vals = []
    # for vec in logicals:
    #     val = (vec.dag() * qt.ptrace(states[-1], [0,1,2]) * vec)[0][0][0]
    #     vals.append(val)
    # errors_after_correction = 1 - np.abs(sum(vals))

    fid2 = np.abs(neilson_fid(rho=qt.ptrace(states[-1], [0,1,2]), sigma=qt.ptrace(ref_state, [0,1,2])))

    return fid1, detectable_error, uncorrectable_error, fid2


import multiprocessing as mp
if __name__ == "__main__":
    dim = 3
    N = 5
    cnots, correction_z, hada_layer, x_layer, vectors = gate(dim=dim, N=N) 
    rho_encoded, ref_state = state(alpha=0, beta=1, dim=dim, hada_layer=hada_layer, cnots=[cnots[0], cnots[1]], state_choice="202")

    iterations = 5
    t1_list = np.linspace(.1, 120, iterations)
    cnots = [cnots[2], cnots[3], cnots[4], cnots[5]]


    values = []
    for i in range(iterations):
        values.append([cnots, rho_encoded, ref_state, [t1_list[i], t1_list[i]], dim, N, x_layer, correction_z, hada_layer, vectors])

    # parallelization of the calculation
    with mp.Pool() as pool:
        results = list(pool.map(sim_func, values))
    print("finished parallelization")

    # data organization 
    fid_pre_correction = []
    fid_post_correction = []
    detectable_errors = []
    uncorrectable_errors = []

    for res in results:
        fid_pre_correction.append(res[0])
        fid_post_correction.append(res[3])
        detectable_errors.append(res[1])
        uncorrectable_errors.append(res[2])

    # creating a csv file to store the information 
    import csv


    # Writing the arrays to a CSV file
    with open('3_wave_cnot_serial.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(fid_pre_correction)  
        writer.writerow(fid_post_correction)  
        writer.writerow(detectable_errors)  
        writer.writerow(uncorrectable_errors)  

    print(f"File created: 3_wave_cnot_serial.csv")




