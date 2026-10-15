#import heapq
import numpy as np
#import cirq
#import cirq_google
#import qsimcirq
#import networkx as nx
import matplotlib.pyplot as plt
#from qutip import about, basis, tensor, destroy, mcsolve, mesolve, expect, qeye, sigmax, sigmay, sigmaz, fock, wigner, coherent
#from scipy.optimize import minimize_scalar
import quimb.tensor as qtn
import os
import time
from matplotlib.backends.backend_pdf import PdfPages


def TNHamiltonian(L, N, D, spin_interaction_coefficient, Spin_Boson_Interaction_Coefficent):

    a = np.zeros((N, N), dtype=complex)
    for n in range(N - 1):
        a[n, n + 1] = np.sqrt(n + 1)
    n_op = np.diag(np.arange(N)).astype(complex)
    proj_top = np.zeros((N, N), dtype=complex)
    proj_top[N - 1, N - 1] = 1.0
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    identity_2 = np.eye(2, dtype=complex)
    identity_N = np.eye(N, dtype=complex)
    op = dict(sx=np.kron(sigma_x, identity_N), sy=np.kron(sigma_y, identity_N), sz=np.kron(sigma_z, identity_N), a=np.kron(identity_2, a), adag=np.kron(identity_2, a.conj().T), n=np.kron(identity_2, n_op), proj_top=np.kron(identity_2, proj_top))

    H1 = {}
    for j in range(L):
        H1[j] = D * (op["a"] + op["adag"]) + Spin_Boson_Interaction_Coefficent * (op["proj_top"] @ op["sx"])

    H2 = {}
    xy_bond = 0.5 * (np.kron(op["sx"], op["sx"]) + np.kron(op["sy"], op["sy"]))
    for j in range(L - 1):
        H2[(j, j + 1)] = spin_interaction_coefficient * xy_bond

    return H1, H2, op


def TNSim(L, N, D, spin_interaction_coefficient, Spin_Boson_Interaction_Coefficent, Total_time, max_bond, dt, tebd_order, steps_per_sample):
    n_samples = int(Total_time / (steps_per_sample * dt) + 1e-9) + 1
    times = np.arange(n_samples) * steps_per_sample * dt

    d = 2 * N
    vec0 = np.zeros(d, dtype=complex)
    vec0[0] = 1.0
    psi0 = qtn.MPS_product_state(vec0.copy() for _ in range(L))

    H1, H2, op = TNHamiltonian(L, N, D, spin_interaction_coefficient, Spin_Boson_Interaction_Coefficent)
    ham = qtn.LocalHam1D(L, H2=H2, H1=H1)
    tebd = qtn.TEBD(psi0, ham, dt=dt, split_opts=({} if max_bond is None else dict(max_bond=max_bond)))

    occupation = np.zeros((len(times), L))
    magnetization = np.zeros((len(times), L))
    truncation_error = np.zeros(len(times))
    trotter_error = np.zeros(len(times))
    for it in range(len(times)):
        if it > 0:
            for step in range(steps_per_sample):
                tebd.step(order=tebd_order, queue=(step < steps_per_sample - 1)) # Queued steps let quimb merge adjacent half-sweeps, the last step flushes the queue so psi is fully evolved before measuring
        psi = tebd.pt
        norm = np.real(psi.H @ psi) # Truncated weight is not renormalised, so 1 - <psi|psi> is the cumulative truncation error
        for j in range(L):
            occupation[it, j] = np.real(psi.local_expectation_exact(op["n"], (j,))) / norm
            magnetization[it, j] = np.real(psi.local_expectation_exact(op["sz"], (j,))) / norm
        truncation_error[it] = 1 - norm
        trotter_error[it] = tebd.err # quimb's TEBD.err is a Trotter error estimate (||H|| * dt^(order+1) per step), not truncation

    return times, occupation, magnetization, truncation_error, trotter_error

def save_data(Scan_Results, Data_Path, Number_of_Bosonic_Modes, Number_of_Fock_States, Total_time, Timestep, Steps_per_sample, max_bond, tebd_order):
    """Overwrite the raw data file with all completed J runs. Arrays are indexed [J, time, site] in the order of the J array."""
    J, times, occupation, magnetization, truncation_error, trotter_error, run_time = (np.array(column) for column in zip(*Scan_Results))
    tmp_path = Data_Path + '.tmp'
    with open(tmp_path, 'wb') as f: # Pass a file handle as np.savez would otherwise append .npz to the temp name
        np.savez(f, J=J, times=times[0], occupation=occupation, magnetization=magnetization, truncation_error=truncation_error, trotter_error=trotter_error, run_time=run_time,
                 L=Number_of_Bosonic_Modes, N=Number_of_Fock_States,
                 Total_time=Total_time, Timestep=Timestep, Steps_per_sample=Steps_per_sample, max_bond=max_bond, tebd_order=tebd_order)
    os.replace(tmp_path, Data_Path)

def save_results(Scan_Results, Output_Path, Number_of_Bosonic_Modes, Number_of_Fock_States, Total_time, Timestep, Steps_per_sample, max_bond, tebd_order):
    """Plot all completed J runs and overwrite the output PDF (written to a temp file first so an interrupted save never corrupts the last good PDF)."""
    fig, axes = plt.subplots(len(Scan_Results), 2, figsize=(11, 3.5*len(Scan_Results)), squeeze=False, layout="constrained")
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, Number_of_Bosonic_Modes))
    for row, (J, times, occupation, magnetization, truncation_error, trotter_error, run_time) in enumerate(Scan_Results):
        ax_occ, ax_mag = axes[row]
        for j in range(Number_of_Bosonic_Modes):
            ax_occ.plot(times, occupation[:, j], color=colors[j], lw=1.8, label=f"site {j}")
            ax_mag.plot(times, magnetization[:, j], color=colors[j], lw=1.8)
        ax_occ.set_xlabel("t"); ax_occ.set_ylabel("<n_j>"); ax_occ.set_title(f"J={J}, Boson occupation (run time {run_time:.1f}s)", fontsize=10)
        ax_mag.set_xlabel("t"); ax_mag.set_ylabel("<sz_j>"); ax_mag.set_title("Spin magnetization")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_T{Total_time}_Bond{max_bond}_Step{Timestep}_K{Steps_per_sample}_TEBD{tebd_order}")

    tmp_path = Output_Path + '.tmp'
    with PdfPages(tmp_path) as pdf:
        pdf.savefig(fig)
    os.replace(tmp_path, Output_Path)
    plt.close(fig)

def run_J(J, Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, Spin_Boson_Interaction_Coefficent, Total_time, max_bond, Timestep, tebd_order, Steps_per_sample):
    """Run one J value in a worker process."""
    t0 = time.time()
    times, occupation, magnetization, truncation_error, trotter_error = TNSim(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, J, Spin_Boson_Interaction_Coefficent, Total_time, max_bond, Timestep, tebd_order, Steps_per_sample)
    return J, times, occupation, magnetization, truncation_error, trotter_error, time.time() - t0