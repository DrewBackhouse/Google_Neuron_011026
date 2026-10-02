import sys
import time
import numpy as np
import cirq
import cirq_google
import qsimcirq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from qutip import basis, tensor, mesolve, expect, fock

from Functions import TrotterStepCRZ, QutipHamiltonian, MapQubitsToDevice, PlotQubitEmbedding, UnaryPostSelection

#---Model Parameters---#

Number_of_Fock_States = 10
Number_of_Bosonic_Modes = 1
Displacement_Coefficent = 1
Spin_Interaction_Coefficent = 0.1
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip

#---Simulation Parameters---#

Total_time = 16
Timesteps_List = list(range(20, 40))
Number_of_Shots = 2000
Noise = True
print(f'Noise = {Noise}')

#---Qutip Sim (independent of Timesteps since Total_time is fixed)---#

state_list = [tensor(basis(2,0),fock(Number_of_Fock_States,0)) for _ in range(Number_of_Bosonic_Modes)]
psi0 = tensor(state_list)
Qutip_Time_Data = np.linspace(0, Total_time, int(Total_time*10))
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
    qsim_options = qsimcirq.QSimOptions(cpu_threads=4, max_fused_gate_size=3)
    sim = qsimcirq.QSimSimulator(qsim_options, noise=noise_model)
    device = cirq_google.engine.create_device_from_processor_id(processor_id)
    cal = cirq_google.engine.load_median_device_calibration(processor_id)
    sim_processor = cirq_google.engine.SimulatedLocalProcessor(processor_id=processor_id, sampler=sim, device=device, calibrations={cal.timestamp // 1000: cal})
    sim_engine = cirq_google.engine.SimulatedLocalEngine([sim_processor])
    simulator = sim_engine.get_sampler(processor_id)
else:
    simulator = cirq.Simulator()

#---Qubit mapping (computed once; interaction graph is independent of Trotter time)---#

if Noise == True:
    Ref_circuit, qubits = TrotterStepCRZ(Number_of_Fock_States, Number_of_Bosonic_Modes, Total_time/Timesteps_List[0], Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
    Willow_qubits = MapQubitsToDevice(qubits, Ref_circuit, device, cal)
    Qubit_Map = dict(zip(qubits, Willow_qubits))
    Mapping_fig = PlotQubitEmbedding(cal, Willow_qubits, Ref_circuit.transform_qubits(Qubit_Map), Number_of_Fock_States, Number_of_Bosonic_Modes)
    Mapping_fig.savefig('Qubit_Mapping.png', dpi=80)

    print('Qubit mapping:')
    for k, (q, w) in enumerate(Qubit_Map.items()):
        if k < Number_of_Bosonic_Modes*Number_of_Fock_States:
            role = f'boson {k//Number_of_Fock_States}, Fock {k%Number_of_Fock_States}'
        else:
            role = f'spin {k-Number_of_Bosonic_Modes*Number_of_Fock_States}'
        print(f'  {q} [{role}] -> {w}')
    print('Mapping plot saved to Qubit_Mapping.png')
    if input('Proceed with this mapping? [y/N] ').strip().lower() not in ('y', 'yes'):
        print('Aborted')
        sys.exit()

#---Scan over Timesteps---#

Scan_Results = []
for Timesteps in Timesteps_List:
    start = time.time()
    Time = Total_time/Timesteps

    Trotter_circuit, qubits = TrotterStepCRZ(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
    if Noise == True:
        Trotter_circuit = Trotter_circuit.transform_qubits(Qubit_Map)
        qubits = Willow_qubits
    Trotter_circuit = cirq.optimize_for_target_gateset(Trotter_circuit, gateset=cirq.CZTargetGateset())

    All_Results = []
    Min_Kept = Number_of_Shots
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
        Min_Kept = min(Min_Kept, len(Full_Results_Postselected))
        Averaged_Results = Full_Results_Postselected.mean(axis=0)
        Boson_Results = Averaged_Results[:Number_of_Bosonic_Modes*Number_of_Fock_States].reshape(Number_of_Bosonic_Modes, Number_of_Fock_States)
        Boson_Occupation_Number = np.sum(Boson_Results * np.arange(0, Number_of_Fock_States, 1), axis=1)
        Timestep_Results = np.concatenate([Boson_Occupation_Number, Averaged_Results[Number_of_Bosonic_Modes*Number_of_Fock_States:]])
        All_Results.append(Timestep_Results)

    Time_Data = np.linspace(Time, Time*Timesteps, Timesteps)
    Scan_Results.append((Timesteps, Time, Time_Data, np.array(All_Results)))
    print(f'Timesteps={Timesteps} complete in {time.time()-start:.1f}s, min {Min_Kept}/{Number_of_Shots} shots kept')

#---Plots---#

Rows = len(Timesteps_List)
fig, axes = plt.subplots(Rows, 2, figsize=(12, 2.8*Rows), squeeze=False)
for r, (Timesteps, Time, Time_Data, All_Results) in enumerate(Scan_Results):
    ax_b, ax_s = axes[r]
    for i in range(Number_of_Bosonic_Modes):
        ax_b.plot(Qutip_Time_Data, exp_n[i])
        ax_b.scatter(Time_Data, All_Results[:,i], s=12, label=f'Mode {i}')
        ax_s.plot(Qutip_Time_Data, -(exp_sz[i]-1)/2)
        ax_s.scatter(Time_Data, All_Results[:,i+Number_of_Bosonic_Modes], s=12, label=f'Spin {i}')
    ax_b.set_title(f'Timesteps={Timesteps}, Trotter time={Time:.3f}')
    ax_s.set_title(f'Timesteps={Timesteps}, Trotter time={Time:.3f}')
    ax_b.set_ylabel('Avg boson occupation')
    ax_s.set_ylabel('Avg spin state')
    ax_b.set_ylim(-0.2, Number_of_Fock_States-0.8)
    ax_s.set_ylim(-0.05, 1.05)
    ax_b.legend(loc='upper right')
    ax_s.legend(loc='upper right')
axes[-1,0].set_xlabel('Time')
axes[-1,1].set_xlabel('Time')
Fig_Height = 2.8*Rows     # inches; keep the suptitle a fixed distance from the top regardless of row count
fig.suptitle(f'N={Number_of_Fock_States}, L={Number_of_Bosonic_Modes}, Total time={Total_time}, Shots={Number_of_Shots}, Noise = {Noise}, postselection', y=1 - 0.15/Fig_Height, va='top')
fig.tight_layout(rect=(0, 0, 1, 1 - 0.6/Fig_Height))
# fig.savefig('Timestep_Scan.png', dpi=80)
with PdfPages(f'N{Number_of_Fock_States} CRZ_Noisy_Timestep_Scan.pdf') as pdf:
    if Noise == True:
        pdf.savefig(Mapping_fig)
    pdf.savefig(fig)
print(f'Saved N{Number_of_Fock_States} CRZ_Noisy_Timestep_Scan.pdf')
