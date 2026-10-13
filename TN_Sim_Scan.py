import os
import sys
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from Functions import TNHamiltonian, TNSim

#---Model Parameters---#

Number_of_Fock_States = 8
Number_of_Bosonic_Modes = 3
Displacement_Coefficent = 1
J_List = np.round(np.arange(0.1, 1.1 + 1e-9, 0.1), 6)
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip

#---Simulation Parameters (defaults, overridable from the command line for cluster array jobs)---#

Total_time = 20
Timestep = 0.1
Steps_per_sample = round(0.5/Timestep)
max_bond = 64
tebd_order = 2

parser = argparse.ArgumentParser()
parser.add_argument('--max_bond', type=int, default=max_bond)
parser.add_argument('--timestep', type=float, default=Timestep)
parser.add_argument('--tebd_order', type=int, default=tebd_order, choices=[2, 4])
parser.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())), help='Number of J values run in parallel')
args = parser.parse_args()
max_bond, Timestep, tebd_order = args.max_bond, args.timestep, args.tebd_order

#---Output---#

os.makedirs('Output', exist_ok=True)
Output_Path = f'Output/TN_Sim_Scan_L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_T{Total_time}_Bond{max_bond}_Step{Timestep}_K{Steps_per_sample}_TEBD{tebd_order}.pdf'

def save_results(Scan_Results):
    """Plot all completed J runs and overwrite the output PDF (written to a temp file first so an interrupted save never corrupts the last good PDF)."""
    fig, axes = plt.subplots(len(Scan_Results), 2, figsize=(11, 3.5*len(Scan_Results)), squeeze=False, layout="constrained")
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, Number_of_Bosonic_Modes))
    for row, (J, times, occupation, magnetization, truncation_error, trotter_error) in enumerate(Scan_Results):
        ax_occ, ax_mag = axes[row]
        for j in range(Number_of_Bosonic_Modes):
            ax_occ.plot(times, occupation[:, j], color=colors[j], lw=1.8, label=f"site {j}")
            ax_mag.plot(times, magnetization[:, j], color=colors[j], lw=1.8)
        ax_occ.set_xlabel("t"); ax_occ.set_ylabel("<n_j>"); ax_occ.set_title(f"J={J}, Boson occupation", fontsize=10)
        ax_mag.set_xlabel("t"); ax_mag.set_ylabel("<sz_j>"); ax_mag.set_title("Spin magnetization")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_T{Total_time}_Bond{max_bond}_Step{Timestep}_K{Steps_per_sample}_TEBD{tebd_order}")

    tmp_path = Output_Path + '.tmp'
    with PdfPages(tmp_path) as pdf:
        pdf.savefig(fig)
    os.replace(tmp_path, Output_Path)
    plt.close(fig)

def run_J(J):
    """Run one J value in a worker process."""
    t0 = time.time()
    times, occupation, magnetization, truncation_error, trotter_error = TNSim(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, J, Spin_Boson_Interaction_Coefficent, Total_time, max_bond, Timestep, tebd_order, Steps_per_sample)
    return J, times, occupation, magnetization, truncation_error, trotter_error, time.time() - t0

#---Scan over J (in parallel)---#

if __name__ == '__main__':
    workers = max(1, min(args.workers, len(J_List)))
    print(f"Running {len(J_List)} J values on {workers} workers: max_bond={max_bond}, Timestep={Timestep}, tebd_order={tebd_order}, steps_per_sample={Steps_per_sample}", flush=True)
    Scan_Results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_J, J) for J in J_List]
        for future in as_completed(futures):
            J, times, occupation, magnetization, truncation_error, trotter_error, tn_time = future.result()
            print(f"J={J}: TEBD (max_bond={max_bond}) done in {tn_time:.2f}s, final truncation error={truncation_error[-1]:.2e}, Trotter error={trotter_error[-1]:.2e}", flush=True)
            Scan_Results.append((J, times, occupation, magnetization, truncation_error, trotter_error))
            Scan_Results.sort(key=lambda result: result[0]) # Runs finish out of order, keep plots ordered by J
            save_results(Scan_Results)
            print(f'Saved {Output_Path}', flush=True)
