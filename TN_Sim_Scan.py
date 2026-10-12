import os
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from Functions import TNHamiltonian, TNSim

#---Model Parameters---#

Number_of_Fock_States = 8
Number_of_Bosonic_Modes = 5
Displacement_Coefficent = 1
J_List = np.arange(0.1, 1.5, 0.1)
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip

#---Simulation Parameters---#

Total_time = 50
Timepoints = 100
max_bond = 128
Timestep = 0.02
tebd_order = 4

#---Output---#

os.makedirs('Output', exist_ok=True)
Output_Path = f'Output/TN_Sim_Scan_L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_T{Total_time}.pdf'

def save_results(Scan_Results):
    """Plot all completed J runs and overwrite the output PDF (written to a temp file first so an interrupted save never corrupts the last good PDF)."""
    fig, axes = plt.subplots(len(Scan_Results), 2, figsize=(11, 3.5*len(Scan_Results)), squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, Number_of_Bosonic_Modes))
    for row, (J, occupation, magnetization, truncation_error, trotter_error) in enumerate(Scan_Results):
        ax_occ, ax_mag = axes[row]
        for j in range(Number_of_Bosonic_Modes):
            ax_occ.plot(times, occupation[:, j], color=colors[j], lw=1.8, label=f"site {j}")
            ax_mag.plot(times, magnetization[:, j], color=colors[j], lw=1.8)
        ax_occ.set_xlabel("t"); ax_occ.set_ylabel("<n_j>"); ax_occ.set_title(f"J={J}, Boson occupation, trunc err={truncation_error[-1]:.2e}, Trotter err={trotter_error[-1]:.2e}", fontsize=10)
        ax_mag.set_xlabel("t"); ax_mag.set_ylabel("<sz_j>"); ax_mag.set_title("Spin magnetization")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"TEBD J scan, L={Number_of_Bosonic_Modes} N={Number_of_Fock_States}, max bond={max_bond} ({len(Scan_Results)}/{len(J_List)} J values complete)")
    fig.tight_layout()

    tmp_path = Output_Path + '.tmp'
    with PdfPages(tmp_path) as pdf:
        pdf.savefig(fig)
    os.replace(tmp_path, Output_Path)
    plt.close(fig)

#---Scan over J---#

times = np.linspace(0, Total_time, Timepoints)
Scan_Results = []
for J in J_List:
    t0 = time.time()
    occupation, magnetization, truncation_error, trotter_error = TNSim(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, J, Spin_Boson_Interaction_Coefficent, times, max_bond, Timestep, tebd_order, progbar=False)
    tn_time = time.time() - t0
    print(f"J={J}: TEBD (max_bond={max_bond}) done in {tn_time:.2f}s, final truncation error={truncation_error[-1]:.2e}, Trotter error={trotter_error[-1]:.2e}")
    Scan_Results.append((J, occupation, magnetization, truncation_error, trotter_error))
    save_results(Scan_Results)
    print(f'Saved {Output_Path} ({len(Scan_Results)}/{len(J_List)})')
