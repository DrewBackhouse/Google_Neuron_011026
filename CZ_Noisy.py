import sys
import numpy as np
import cirq
import cirq_google
import qsimcirq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from qutip import about, basis, tensor, destroy, mcsolve, mesolve, expect, qeye, sigmax, sigmay, sigmaz, fock, wigner, coherent

from Functions import TrotterStepCRZ, QutipHamiltonian, MapQubitsToDevice, CheckQubitMapping, PlotQubitEmbedding, UnaryPostSelection, TrotterStepCZ_New

#---Model Parameters---#

Number_of_Fock_States = 8
Number_of_Bosonic_Modes = 1
Displacement_Coefficent = 1 # Independnet of the physics (as far as I am aware, at least when it is global)
Spin_Interaction_Coefficent = 0.1
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip

#---Simulation Parameters---#

Time= np.pi/(2*Displacement_Coefficent*(Number_of_Fock_States-0.5)**0.5) # Trotter step time s.t. the controlled RZ gate becomes a CZ
Timesteps=22
print(Time)
Number_of_Shots = 2000
Noise = True
print(f'Noise = {Noise}')
Simulation_Approval = True

#---Qubit mapping---#

Automatic_Qubit_Mapping = True
Manual_Qubit_Mapping = [cirq.GridQubit(6, 1), cirq.GridQubit(6, 2), cirq.GridQubit(5, 2), cirq.GridQubit(4, 2), cirq.GridQubit(4, 3), cirq.GridQubit(3, 3), cirq.GridQubit(3, 4), cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)] # Used if Automatic_Qubit_Mapping = False. Boson qubits (mode by mode, Fock 0..N-1) then spin qubits
Boson_Weights = {'T1': 1.0, 'Tphi': 1.0, 'single_qubit': 1.0, 'readout': 1.0, 'CZ': 1.0, 'coherent': 1.0}  # Used if Automatic_Qubit_Mapping = True
Spin_Weights = {'T1': 1.0, 'Tphi': 5.0, 'single_qubit': 1.0, 'readout': 1.0, 'CZ': 1.0, 'coherent': 1.0}   # Spin weights also apply to the spin-boson couplers
print(f'Automatic_Qubit_Mapping = {Automatic_Qubit_Mapping}')


#---Qutip Sim---#

Qutip_Time = Time * Timesteps
state_list = [tensor(basis(2,0),fock(Number_of_Fock_States,0)) for _ in range(Number_of_Bosonic_Modes)]       # Tensor product of spin and boson vectors in ground state at each lattice point in a list 
psi0 = tensor(state_list)                                           # Tensor product of all entries in the list
Qutip_Time_Data = np.linspace(0, Qutip_Time, int(Qutip_Time*10))
H, n_list, sz_list = QutipHamiltonian(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
result = mesolve(H, psi0, Qutip_Time_Data, args={'Displacement_Coefficent': Displacement_Coefficent, 'Spin_Interaction_Coefficent': Spin_Interaction_Coefficent, 'Spin_Boson_Interaction_Coefficent': Spin_Boson_Interaction_Coefficent})
states = result.states
exp_n = np.array([expect(n_list[i], states) for i in range(Number_of_Bosonic_Modes)])
exp_sz = np.array([expect(sz_list[i], states) for i in range(Number_of_Bosonic_Modes)])

#---Noise model---#

if Noise == True:   
    processor_id = "willow_pink"
    noise_props = cirq_google.engine.load_device_noise_properties(processor_id)
    noise_model = cirq_google.NoiseModelFromGoogleNoiseProperties(noise_props)
    qsim_options = qsimcirq.QSimOptions(cpu_threads=4, max_fused_gate_size=3)       # Fastest measured on this machine (default is 1 thread, fusion 2)
    sim = qsimcirq.QSimSimulator(qsim_options, noise=noise_model)
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    cal = cirq_google.engine.load_median_device_calibration(processor_id)
    sim_processor = cirq_google.engine.SimulatedLocalProcessor(processor_id=processor_id, sampler=sim, device=device, calibrations={cal.timestamp // 1000: cal})
    sim_engine = cirq_google.engine.SimulatedLocalEngine([sim_processor])
    simulator = sim_engine.get_sampler(processor_id)
else:
    simulator = cirq.Simulator()

#---Trotter step circuit preperation---#

Trotter_circuit, qubits = TrotterStepCZ_New(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
print('Trotter step circuit')
print(Trotter_circuit)
if Noise == True:
    if Automatic_Qubit_Mapping == True:
        Willow_qubits = MapQubitsToDevice(qubits, Trotter_circuit, device, cal, qubits[Number_of_Bosonic_Modes*Number_of_Fock_States:], noise_props=noise_props, boson_weights=Boson_Weights, spin_weights=Spin_Weights)
    else:
        if len(Manual_Qubit_Mapping) != len(qubits):
            raise ValueError(f'Manual_Qubit_Mapping has {len(Manual_Qubit_Mapping)} qubits, circuit needs {len(qubits)}')
        Willow_qubits = Manual_Qubit_Mapping
    Trotter_circuit = Trotter_circuit.transform_qubits(dict(zip(qubits, Willow_qubits)))
    CheckQubitMapping(Willow_qubits, Trotter_circuit, device)
    Mapping_fig = PlotQubitEmbedding(cal, Willow_qubits, Trotter_circuit, Number_of_Fock_States, Number_of_Bosonic_Modes, noise_props=noise_props)
    Mapping_fig.savefig('Output/Qubit_Mapping.png', dpi=80)
    qubits = Willow_qubits
Trotter_circuit = cirq.optimize_for_target_gateset(Trotter_circuit, gateset=cirq.CZTargetGateset())
print('Trotter step circuit mapped to native gates')
print(Trotter_circuit.to_text_diagram(qubit_order=qubits))

if Simulation_Approval == True:
    if input('Proceed with this mapping? [y/N] ').strip().lower() not in ('y', 'yes'):
            print('Aborted')
            sys.exit()

#---Cirq Sim---#

All_Results = []
for i in range(1, Timesteps+1):
    circuit = cirq.Circuit()
    for j in range(Number_of_Bosonic_Modes):
        circuit.append(cirq.X(qubits[j*Number_of_Fock_States]))
    for j in range(Number_of_Bosonic_Modes):
            circuit.append(cirq.H(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j]))
    circuit.append([Trotter_circuit]*i)
    for j in range(Number_of_Bosonic_Modes):
                circuit.append(cirq.H(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j]))
    circuit.append(cirq.measure(*qubits, key='m'))
    Full_Results = simulator.run(circuit, repetitions=Number_of_Shots).measurements['m']
    Full_Results_Postselected = UnaryPostSelection(Full_Results, Number_of_Fock_States, Number_of_Bosonic_Modes)
    Averaged_Results = Full_Results_Postselected.mean(axis=0)
    Boson_Results = Averaged_Results[:Number_of_Bosonic_Modes*Number_of_Fock_States].reshape(Number_of_Bosonic_Modes, Number_of_Fock_States)
    Boson_Occupation_Number = np.sum(Boson_Results * np.arange(0, Number_of_Fock_States, 1), axis=1)
    Timestep_Results = np.concatenate([Boson_Occupation_Number, Averaged_Results[Number_of_Bosonic_Modes*Number_of_Fock_States:]])
    All_Results.append(Timestep_Results)
    print(f'timestep {i} complete, {len(Full_Results_Postselected)}/{Number_of_Shots} shots kept')

All_Results = np.array(All_Results)
Time_Data = np.linspace(Time, Time*Timesteps, Timesteps)

#---Plots---#

Fig_Height = 4.0     # inches; gives the same axes height as a row of the timestep scan (suptitle/xlabel overhead ~1.9in)
fig, (ax_b, ax_s) = plt.subplots(1, 2, figsize=(12, Fig_Height))
for i in range(Number_of_Bosonic_Modes):
    ax_b.plot(Qutip_Time_Data, exp_n[i])
    ax_b.scatter(Time_Data, All_Results[:,i], s=12, label=f'Mode {i}')
    ax_s.plot(Qutip_Time_Data, -(exp_sz[i]-1)/2)
    ax_s.scatter(Time_Data, All_Results[:,i+Number_of_Bosonic_Modes], s=12, label=f'Spin {i}')
ax_b.set_title(f'Timesteps={Timesteps}, Trotter time={Time:.3f}')
ax_s.set_title(f'Timesteps={Timesteps}, Trotter time={Time:.3f}')
ax_b.set_ylabel('Avg boson occupation')
ax_s.set_ylabel('Avg spin state')
ax_b.set_xlabel('Time')
ax_s.set_xlabel('Time')
ax_b.set_ylim(-0.2, Number_of_Fock_States-0.8)
ax_s.set_ylim(-0.05, 1.05)
ax_b.legend(loc='upper right')
ax_s.legend(loc='upper right')
fig.suptitle(f'CZ spin-boson interaction, N={Number_of_Fock_States}, L={Number_of_Bosonic_Modes}, Total time={Time*Timesteps:.2f}, Shots={Number_of_Shots}, Noise = {Noise}, postselection', y=1 - 0.15/Fig_Height, va='top')
fig.tight_layout(rect=(0, 0, 1, 1 - 0.6/Fig_Height))
with PdfPages(f'Output/CZ_N{Number_of_Fock_States}_T{Timesteps}_Noise{Noise}.pdf') as pdf:
    if Noise == True:
        pdf.savefig(Mapping_fig)
    pdf.savefig(fig)
print(f'Saved CZ_N{Number_of_Fock_States}_T{Timesteps}_Noise{Noise}.pdf')