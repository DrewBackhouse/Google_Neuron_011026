import heapq
import numpy as np
import cirq
import networkx as nx
import matplotlib.pyplot as plt
from qutip import about, basis, tensor, destroy, mcsolve, mesolve, expect, qeye, sigmax, sigmay, sigmaz, fock, wigner, coherent
from scipy.optimize import minimize_scalar

def QutipHamiltonian(NumberOfBosonicModes, NumberOfFockStates, displacement_coefficient, spin_interaction_coefficient, SpinBosonInteractionCoefficent):
    
    sx_list, sy_list, sz_list = [], [], []                       # Define pauli X and Y operator lists 
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]       # Define list of 2D and ND identity operators for each lattice site
        op_list[i] = tensor(sigmax(), qeye(NumberOfFockStates))                      # Replace ith entry with pauli X and I_N
        sx_list.append(tensor(op_list))                             # Add to Pauli X operator list
        op_list[i] = tensor(sigmay(), qeye(NumberOfFockStates))                      # Repeat for Pauli Y
        sy_list.append(tensor(op_list))
        op_list[i] = tensor(sigmaz(), qeye(NumberOfFockStates))                      # Repeat for Pauli Z
        sz_list.append(tensor(op_list))

    a_list = []                                     # Repeat for ladder and number operators
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), destroy(NumberOfFockStates))
        a_list.append(tensor(op_list))

    n_list = []
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), destroy(NumberOfFockStates).dag() * destroy(NumberOfFockStates))
        n_list.append(tensor(op_list))

    n_max_list = []                                 # Treat |n*><n*| as an operator and repeat
    for i in range(NumberOfBosonicModes):
        op_list = [tensor(qeye(2),qeye(NumberOfFockStates)) for _ in range(NumberOfBosonicModes)]
        op_list[i] = tensor(qeye(2), fock(NumberOfFockStates, NumberOfFockStates-1) * fock(NumberOfFockStates, NumberOfFockStates-1).dag())
        n_max_list.append(tensor(op_list))

    H = sum(displacement_coefficient * (a_list[i] + a_list[i].dag()) + SpinBosonInteractionCoefficent * n_max_list[i] * sx_list[i] for i in range(NumberOfBosonicModes))
    H += sum(spin_interaction_coefficient * (sx_list[i] * sx_list[i+1] + sy_list[i] * sy_list[i+1])/2 for i in range(NumberOfBosonicModes-1))
    
    return H, n_list, sz_list

def QutipSim(Number_of_Fock_States,Number_of_Bosonic_Modes, Total_time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent):
    
    state_list = [tensor(basis(2,0),fock(Number_of_Fock_States,0)) for _ in range(Number_of_Bosonic_Modes)]
    psi0 = tensor(state_list)
    Qutip_Time_Data = np.linspace(0, Total_time, int(Total_time*10))
    H, n_list, sz_list = QutipHamiltonian(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent)
    result = mesolve(H, psi0, Qutip_Time_Data, args={'Displacement_Coefficent': Displacement_Coefficent, 'Spin_Interaction_Coefficent': Spin_Interaction_Coefficent, 'Spin_Boson_Interaction_Coefficent': Spin_Boson_Interaction_Coefficent})
    states = result.states
    exp_n = np.array([expect(n_list[i], states) for i in range(Number_of_Bosonic_Modes)])
    exp_sz = np.array([expect(sz_list[i], states) for i in range(Number_of_Bosonic_Modes)])

    return exp_n, exp_sz, Qutip_Time_Data


def TrotterStepCRZ(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent):
    qubits = cirq.LineQubit.range(Number_of_Bosonic_Modes * (Number_of_Fock_States+1))
    Trotter_circuit = cirq.Circuit()

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

        for i in range(1, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(1, Number_of_Bosonic_Modes-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range (Number_of_Bosonic_Modes):
         Trotter_circuit.append(cirq.rz(2*Time*Spin_Boson_Interaction_Coefficent).controlled()(qubits[(j+1)*Number_of_Fock_States-1],qubits[Number_of_Fock_States*Number_of_Bosonic_Modes+j]))

    return Trotter_circuit, qubits

def TrotterStepCRX(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent):
    qubits = cirq.LineQubit.range(Number_of_Bosonic_Modes * (Number_of_Fock_States+1))
    Trotter_circuit = cirq.Circuit()

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

        for i in range(1, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(1, Number_of_Bosonic_Modes-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range (Number_of_Bosonic_Modes):
         Trotter_circuit.append(cirq.rx(2*Time*Spin_Boson_Interaction_Coefficent).controlled()(qubits[(j+1)*Number_of_Fock_States-1],qubits[Number_of_Fock_States*Number_of_Bosonic_Modes+j]))

    return Trotter_circuit, qubits

def TrotterStepCZ_New(Number_of_Fock_States, Number_of_Bosonic_Modes, Time, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent):
    qubits = cirq.LineQubit.range(Number_of_Bosonic_Modes * (Number_of_Fock_States+1))
    Trotter_circuit = cirq.Circuit()

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

        for i in range(1, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Displacement_Coefficent*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(1, Number_of_Bosonic_Modes-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Spin_Interaction_Coefficent*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range (Number_of_Bosonic_Modes):
         Trotter_circuit.append(cirq.CZ(qubits[(j+1)*Number_of_Fock_States-1],qubits[Number_of_Fock_States*Number_of_Bosonic_Modes+j]))
         Trotter_circuit.append(cirq.Z(qubits[(j+1)*Number_of_Fock_States-1])**-0.5)     # CZ * S^dagger(control) == controlled rz(pi)

    return Trotter_circuit, qubits

ERROR_TYPES = ('T1', 'Tphi', 'single_qubit', 'readout', 'CZ', 'coherent')

def MapQubitsToDevice(qubits, circuit, device, cal, spin_qubits, noise_props=None, boson_weights=None, spin_weights=None, beam_width=2000):
    """Maps logical qubits onto device GridQubits so every 2-qubit interaction in circuit sits on a physical
    coupler (no SWAPs), choosing the layout with the lowest cost = sum over used qubits of median_T1/T1
    + sum over used couplers of CZ_error/median_CZ_error (each weighted). Found by beam search.
    If noise_props is given, also adds per qubit median_Tphi/Tphi, single-qubit gate Pauli error/median and
    (p(0->1)+p(1->0))/median readout error, and per coupler (|phi|+|theta|)/median, where phi, theta are the
    coherent CZ errors from noise_props.fsim_errors (not captured by the XEB Pauli error).
    boson_weights and spin_weights are dicts with keys in ERROR_TYPES (missing keys default to 1.0) weighting each
    error type separately for the logical boson qubits and the logical spin qubits (spin_qubits). A coupler is
    weighted with spin_weights if either end is a spin qubit, otherwise with boson_weights.
    Returns qubits_willow, a list of GridQubits in the same order as qubits."""
    weights = {}
    for role, given in (('boson', boson_weights), ('spin', spin_weights)):
        given = given or {}
        unknown = set(given) - set(ERROR_TYPES)
        if unknown:
            raise ValueError(f'Unknown {role} weight(s) {unknown}, expected keys from {ERROR_TYPES}')
        weights[role] = {k: given.get(k, 1.0) for k in ERROR_TYPES}
    spin_qubits = set(spin_qubits)

    logical_graph = nx.Graph()
    logical_graph.add_nodes_from(qubits)
    for op in circuit.all_operations():
        if len(op.qubits) == 2:
            logical_graph.add_edge(*op.qubits)

    t1 = {key[0]: value[0] for key, value in cal['single_qubit_idle_t1_micros'].items()}
    cz = {frozenset(key): value[0] for key, value in cal['two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle'].items()}
    median_t1, median_cz = np.median(list(t1.values())), np.median(list(cz.values()))
    qubit_errors = {'T1': {q: median_t1 / t1[q] for q in t1}}             # Unweighted, median-normalised error components
    edge_errors = {'CZ': {pair: error / median_cz for pair, error in cz.items()}}

    if noise_props is not None:
        median_tphi = np.median(list(noise_props.tphi_ns.values()))
        qubit_errors['Tphi'] = {q: median_tphi / noise_props.tphi_ns.get(q, median_tphi) for q in t1}
        single_qubit = {op_id.qubits[0]: error for op_id, error in noise_props.gate_pauli_errors.items() if op_id.gate_type == cirq.PhasedXZGate}
        median_single_qubit = np.median(list(single_qubit.values()))
        qubit_errors['single_qubit'] = {q: single_qubit.get(q, median_single_qubit) / median_single_qubit for q in t1}
        readout = {q: sum(errors) for q, errors in noise_props.readout_errors.items()}      # p(0->1) + p(1->0)
        median_readout = np.median(list(readout.values()))
        qubit_errors['readout'] = {q: readout.get(q, median_readout) / median_readout for q in t1}
        coherent = {}
        for op_id, fsim in noise_props.fsim_errors.items():
            if op_id.gate_type == cirq.CZPowGate:
                coherent[frozenset(op_id.qubits)] = abs(fsim.phi) + abs(fsim.theta)
        median_coherent = np.median(list(coherent.values()))
        edge_errors['coherent'] = {pair: coherent.get(pair, median_coherent) / median_coherent for pair in cz}

    qubit_cost = {role: {q: sum(w[k] * errors[q] for k, errors in qubit_errors.items()) for q in t1} for role, w in weights.items()}
    edge_cost = {role: {pair: sum(w[k] * errors[pair] for k, errors in edge_errors.items()) for pair in cz} for role, w in weights.items()}

    device_graph = nx.Graph()                                          # Only qubits and couplers with calibration data
    device_graph.add_nodes_from(q for q in device.metadata.qubit_set if q in t1)
    device_graph.add_edges_from(tuple(pair) for pair in device.metadata.qubit_pairs if frozenset(pair) in cz)

    order = []                                                         # BFS order from the centre of each connected component
    for component in sorted(nx.connected_components(logical_graph), key=len, reverse=True):
        root = nx.center(logical_graph.subgraph(component))[0]
        order += [root] + [v for _, v in nx.bfs_edges(logical_graph, root)]
    position = {q: i for i, q in enumerate(order)}
    earlier_neighbours = [[position[n] for n in logical_graph[q] if position[n] < i] for i, q in enumerate(order)]
    last_neighbour = [max([i] + [position[n] for n in logical_graph[q]]) for i, q in enumerate(order)]
    role = ['spin' if q in spin_qubits else 'boson' for q in order]
    edge_role = [['spin' if role[i] == 'spin' or role[j] == 'spin' else 'boson' for j in earlier_neighbours[i]] for i in range(len(order))]

    beam = [(0.0, ())]
    for i in range(len(order)):
        best = {}
        for cost, placement in beam:
            neighbours = [placement[j] for j in earlier_neighbours[i]]
            if neighbours:
                candidates = set.intersection(*(set(device_graph[n]) for n in neighbours))
            else:
                candidates = set(device_graph.nodes)
            for p in candidates.difference(placement):
                new_placement = placement + (p,)
                new_cost = cost + qubit_cost[role[i]][p] + sum(edge_cost[r][frozenset((p, n))] for r, n in zip(edge_role[i], neighbours))
                # Layouts with the same used qubits and same images of still-open logical qubits have identical futures
                key = (frozenset(new_placement), tuple(new_placement[j] for j in range(i+1) if last_neighbour[j] > i))
                if key not in best or new_cost < best[key][0]:
                    best[key] = (new_cost, new_placement)
        beam = heapq.nsmallest(beam_width, best.values())
        if not beam:
            raise ValueError(f'Could not embed the {len(qubits)}-qubit interaction graph onto the device with local interactions only')

    mapping = dict(zip(order, beam[0][1]))
    return [mapping[q] for q in qubits]

def CheckQubitMapping(qubits_willow, circuit, device):
    """Checks a mapping onto the device: qubits_willow must be distinct device qubits and every 2-qubit operation
    in circuit (already transformed onto qubits_willow) must sit on a device coupler. Raises ValueError otherwise."""
    if len(set(qubits_willow)) != len(qubits_willow):
        raise ValueError(f'Qubit mapping uses a device qubit more than once: {qubits_willow}')
    missing = [q for q in qubits_willow if q not in device.metadata.qubit_set]
    if missing:
        raise ValueError(f'Qubit mapping uses qubits not on the device: {missing}')
    couplers = {frozenset(pair) for pair in device.metadata.qubit_pairs}
    non_local = {tuple(op.qubits) for op in circuit.all_operations() if len(op.qubits) == 2 and frozenset(op.qubits) not in couplers}
    if non_local:
        raise ValueError(f'Qubit mapping puts 2-qubit operations on qubits without a coupler: {sorted(non_local)}')

EMBEDDING_COLOR_BOSON = 'tab:red'
EMBEDDING_COLOR_SPIN = 'tab:blue'
EMBEDDING_COLOR_MIXED = 'tab:purple'

def PlotQubitEmbedding(cal, qubits_willow, circuit, Number_of_Fock_States, Number_of_Bosonic_Modes, noise_props=None):
    """Plots the device calibration (T1 per qubit, CZ error per coupler) with the qubits and couplers used by
    the embedding highlighted: boson in blue, spin in red, spin-boson couplers in purple.
    If noise_props is given, adds two more rows plotting Tphi per qubit, the coherent CZ error |phi|+|theta| per
    coupler (not captured by T1 or the XEB Pauli error), single-qubit gate Pauli error and readout error
    p(0->1)+p(1->0) per qubit, and prints those values for the used qubits/couplers.
    circuit must already be on qubits_willow. Returns the Figure."""
    boson_qubits = set(qubits_willow[:Number_of_Bosonic_Modes*Number_of_Fock_States])
    spin_qubits = set(qubits_willow[Number_of_Bosonic_Modes*Number_of_Fock_States:])
    boson_edges, spin_edges, mixed_edges = set(), set(), set()
    for op in circuit.all_operations():
        if len(op.qubits) == 2:
            qa, qb = op.qubits
            if qa in boson_qubits and qb in boson_qubits:
                boson_edges.add((qa, qb))
            elif qa in spin_qubits and qb in spin_qubits:
                spin_edges.add((qa, qb))
            else:
                mixed_edges.add((qa, qb))

    if noise_props is not None:
        single_qubit_pct = {op_id.qubits[0]: error * 100 for op_id, error in noise_props.gate_pauli_errors.items() if op_id.gate_type == cirq.PhasedXZGate}
        readout_pct = {q: sum(errors) * 100 for q, errors in noise_props.readout_errors.items()}      # p(0->1) + p(1->0)
        median_tphi = np.median(list(noise_props.tphi_ns.values()))
        fsim = {frozenset(op_id.qubits): v for op_id, v in noise_props.fsim_errors.items() if op_id.gate_type == cirq.CZPowGate}
        for qa, qb in sorted(boson_edges | spin_edges | mixed_edges):
            v = fsim.get(frozenset((qa, qb)))

    def highlight_qubits(ax):
        for group, color in ((boson_qubits, EMBEDDING_COLOR_BOSON), (spin_qubits, EMBEDDING_COLOR_SPIN)):
            ax.scatter([q.col for q in group], [q.row for q in group], s=280, facecolors='none', edgecolors=color, linewidths=2.5, zorder=5)

    def highlight_couplers(ax):
        for edges, color in ((boson_edges, EMBEDDING_COLOR_BOSON), (spin_edges, EMBEDDING_COLOR_SPIN), (mixed_edges, EMBEDDING_COLOR_MIXED)):
            for qa, qb in edges:
                ax.plot([qa.col, qb.col], [qa.row, qb.row], color=color, linewidth=3.5, zorder=5, solid_capstyle='round')
        for group, color in ((boson_qubits, EMBEDDING_COLOR_BOSON), (spin_qubits, EMBEDDING_COLOR_SPIN)):
            ax.scatter([q.col for q in group], [q.row for q in group], s=45, color=color, zorder=6)

    if noise_props is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    else:
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(18, 27))

    cal.heatmap('single_qubit_idle_t1_micros').plot(ax1)
    highlight_qubits(ax1)

    two_qubit_error_pct = {
        cal.key_to_qubits(key): cal.value_to_float(value) * 100
        for key, value in cal['two_qubit_parallel_cz_gate_xeb_pauli_error_per_cycle'].items()
    }
    cirq.TwoQubitInteractionHeatmap(
        two_qubit_error_pct,
        title='Two Qubit Parallel Cz Gate Xeb Pauli Error Per Cycle',
        annotation_format='.2f',
        annotation_text_kwargs={'fontsize': 7},
        colorbar_options={'label': '%'},
    ).plot(ax2)
    highlight_couplers(ax2)

    if noise_props is not None:
        cirq.Heatmap(
            {(q,): tphi / 1000 for q, tphi in noise_props.tphi_ns.items()},
            title='Single Qubit Tphi Micros',
            annotation_format='.0f',
        ).plot(ax3)
        highlight_qubits(ax3)

        coherent_mrad = {frozenset(op_id.qubits): (abs(v.phi) + abs(v.theta)) * 1000      # Both qubit orderings are listed; keep one
                         for op_id, v in noise_props.fsim_errors.items() if op_id.gate_type == cirq.CZPowGate}
        cirq.TwoQubitInteractionHeatmap(
            {tuple(pair): value for pair, value in coherent_mrad.items()},
            title='Coherent CZ Error |phi| + |theta|',
            annotation_format='.0f',
            annotation_text_kwargs={'fontsize': 7},
            colorbar_options={'label': 'mrad'},
        ).plot(ax4)
        highlight_couplers(ax4)

        cirq.Heatmap(
            {(q,): error for q, error in single_qubit_pct.items()},
            title='Single Qubit Gate Pauli Error',
            annotation_format='.2f',
            colorbar_options={'label': '%'},
        ).plot(ax5)
        highlight_qubits(ax5)

        cirq.Heatmap(
            {(q,): error for q, error in readout_pct.items()},
            title='Readout Error p(0->1) + p(1->0)',
            annotation_format='.1f',
            colorbar_options={'label': '%'},
        ).plot(ax6)
        highlight_qubits(ax6)

    fig.suptitle(f'Calibration ({cal.timestamp_str()}) -- embedding highlighted', fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96 if noise_props is None else 0.985])
    return fig

def UnaryPostSelection(Full_Results, Number_of_Fock_States, Number_of_Bosonic_Modes):
    Boson_Results = Full_Results[:, :Number_of_Bosonic_Modes*Number_of_Fock_States].reshape(-1, Number_of_Bosonic_Modes, Number_of_Fock_States)     # (shots, modes, Fock states)
    Unary_Mask = np.all(Boson_Results.sum(axis=2) == 1, axis=1)          # True if every register has exactly one qubit in |1>
    Full_Results_Postselected = Full_Results[Unary_Mask]
    return Full_Results_Postselected

#---Old Functions---#

def TrotterStep(Number_of_Fock_States, Number_of_Bosonic_Modes, Time):
    qubits = cirq.LineQubit.range(Number_of_Bosonic_Modes * (Number_of_Fock_States+1))
    Trotter_circuit = cirq.Circuit()

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

        for i in range(1, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(1, Number_of_Bosonic_Modes-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range (Number_of_Bosonic_Modes):
         Trotter_circuit.append(cirq.CX(qubits[(j+1)*Number_of_Fock_States-1],qubits[Number_of_Fock_States*Number_of_Bosonic_Modes+j]))

    return Trotter_circuit, qubits

def TrotterStepCZ(Number_of_Fock_States, Number_of_Bosonic_Modes, Time):
    qubits = cirq.LineQubit.range(Number_of_Bosonic_Modes * (Number_of_Fock_States+1))
    Trotter_circuit = cirq.Circuit()

    for j in range(Number_of_Bosonic_Modes):
        for i in range(0, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

        for i in range(1, Number_of_Fock_States-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-2*Time*(i+1)**0.5/np.pi)(qubits[j*Number_of_Fock_States+i], qubits[j*Number_of_Fock_States+i+1]))

    for j in range(0, Number_of_Bosonic_Modes-1, 2):
        Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range(1, Number_of_Bosonic_Modes-1, 2):
            Trotter_circuit.append(cirq.ISwapPowGate(exponent=-4*Time/np.pi)(qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j], qubits[Number_of_Bosonic_Modes*Number_of_Fock_States+j+1]))

    for j in range (Number_of_Bosonic_Modes):
         Trotter_circuit.append(cirq.CZ(qubits[(j+1)*Number_of_Fock_States-1],qubits[Number_of_Fock_States*Number_of_Bosonic_Modes+j]))

    return Trotter_circuit, qubits