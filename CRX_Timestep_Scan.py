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

from Functions import QutipHamiltonian, QutipSim, TrotterStepCRX, MapQubitsToDevice, CheckQubitMapping, PlotQubitEmbedding, UnaryPostSelection

#---Model Parameters---#

Number_of_Fock_States = 8
Number_of_Bosonic_Modes = 1
Displacement_Coefficent = 1
Spin_Interaction_Coefficent = 0.1
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip

#---Simulation Parameters---#

Total_time = 20
Timesteps_List = list(range(5, 6))
Number_of_Shots = 2000
Noise = True
Simulation_Approval = True
Automatic_Qubit_Mapping = True

#---Qubit mapping---#

Manual_Qubit_Mapping = [cirq.GridQubit(6, 1), cirq.GridQubit(6, 2), cirq.GridQubit(5, 2), cirq.GridQubit(4, 2), cirq.GridQubit(4, 3), cirq.GridQubit(3, 3), cirq.GridQubit(3, 4), cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)] # Used if Automatic_Qubit_Mapping = False. Boson qubits (mode by mode, Fock 0..N-1) then spin qubits
Boson_Weights = {'T1': 1.0, 'Tphi': 1.0, 'single_qubit': 1.0, 'readout': 1.0, 'CZ': 2.0, 'coherent': 2.0}  # Used if Automatic_Qubit_Mapping = True
Spin_Weights = {'T1': 1.0, 'Tphi': 1.0, 'single_qubit': 1.0, 'readout': 1.0, 'CZ': 2.0, 'coherent': 2.0}   # Spin weights also apply to the spin-boson couplers

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

#---Trotter circuit and Qubit mapping---#

Trotter_circuit, qubits = TrotterStepCRX(Number_of_Fock_States, Number_of_Bosonic_Modes, Total_time/Timesteps_List[0], Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
print(Trotter_circuit)
Trotter_Diagram = str(Trotter_circuit)

if Noise == True:
    if Automatic_Qubit_Mapping == True:
        Willow_qubits = MapQubitsToDevice(qubits, Trotter_circuit, device, cal, qubits[Number_of_Bosonic_Modes*Number_of_Fock_States:], noise_props=noise_props, boson_weights=Boson_Weights, spin_weights=Spin_Weights)
    else:
        if len(Manual_Qubit_Mapping) != len(qubits):
            raise ValueError(f'Manual_Qubit_Mapping has {len(Manual_Qubit_Mapping)} qubits, circuit needs {len(qubits)}')
        Willow_qubits = Manual_Qubit_Mapping
    Qubit_Map = dict(zip(qubits, Willow_qubits))
    Trotter_circuit = Trotter_circuit.transform_qubits(Qubit_Map)
    CheckQubitMapping(Willow_qubits, Trotter_circuit, device)
    Mapping_fig = PlotQubitEmbedding(cal, Willow_qubits, Trotter_circuit, Number_of_Fock_States, Number_of_Bosonic_Modes, noise_props=noise_props)
    Mapping_fig.savefig('Output/Qubit_Mapping.png', dpi=80)
    Trotter_circuit = cirq.optimize_for_target_gateset(Trotter_circuit, gateset=cirq.CZTargetGateset())
    print(Trotter_circuit)
    print('Mapping plot saved to Qubit_Mapping.png')
    if Simulation_Approval == True:
        if input('Proceed with simulation? [y/N] ').strip().lower() not in ('y', 'yes'):
            print('Aborted')
            sys.exit()

#---Qutip Sim---#

exp_n, exp_sz, Qutip_Time_Data = QutipSim(Number_of_Fock_States,Number_of_Bosonic_Modes, Total_time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)

#---Scan over Timesteps---#

Scan_Results = []
for Timesteps in Timesteps_List:
    start = time.time()
    Time = Total_time/Timesteps

    Trotter_circuit, qubits = TrotterStepCRX(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
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
        circuit.append([Trotter_circuit]*i)
        circuit.append(cirq.measure(*qubits, key='m'))
        if i == 1 and Timesteps == Timesteps_List[0]:
            print(circuit)
            Full_Circuit_Diagram = str(circuit)
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

Circuit_Text = (f'Trotter step circuit (Trotter time={Total_time/Timesteps_List[0]:.3f})\n\n{Trotter_Diagram}\n\n\n'
                f'Full circuit, first timestep (Timesteps={Timesteps_List[0]})\n\n{Full_Circuit_Diagram}')
Lines = Circuit_Text.split('\n')
Font_Size = 6
Circuit_fig = plt.figure(figsize=(max(8, 0.6*Font_Size/72*max(len(l) for l in Lines) + 1), max(4, 1.2*Font_Size/72*len(Lines) + 1)))   # size page to fit the text so wide circuits aren't clipped
Circuit_fig.text(0.5/Circuit_fig.get_figwidth(), 1 - 0.5/Circuit_fig.get_figheight(), Circuit_Text, family='monospace', fontsize=Font_Size, va='top', ha='left')

with PdfPages(f'Output/CRXTimestepScan_N{Number_of_Fock_States}_T{Timesteps_List[0]}-{Timesteps_List[-1]}_Noise{Noise}.pdf') as pdf:
    if Noise == True:
        pdf.savefig(Mapping_fig)
    pdf.savefig(Circuit_fig)
    pdf.savefig(fig)
print(f'Saved Output/CRXTimestepScan_N{Number_of_Fock_States}_T{Timesteps_List[0]}-{Timesteps_List[-1]}_Noise{Noise}.pdf')
