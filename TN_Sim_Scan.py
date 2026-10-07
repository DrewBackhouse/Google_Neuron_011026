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

Number_of_Fock_States = 6
Number_of_Bosonic_Modes = 4
Displacement_Coefficent = 1
J_List = [0.1, 1] # Spin interaction coefficients to scan over
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip

#---Simulation Parameters---#

Total_time = 20
Timepoints = 40
max_bond = 64
Timestep = 0.02
tebd_order = 4

#---Scan over J---#

times = np.linspace(0, Total_time, Timepoints)
Scan_Results = []
for J in J_List:
    t0 = time.time()
    occupation, magnetization, truncation_error, trotter_error = TNSim(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, J, Spin_Boson_Interaction_Coefficent, times, max_bond, Timestep, tebd_order, progbar=False)
    tn_time = time.time() - t0
    print(f"J={J}: TEBD (max_bond={max_bond}) done in {tn_time:.2f}s, final truncation error={truncation_error[-1]:.2e}, Trotter error={trotter_error[-1]:.2e}")
    Scan_Results.append((J, occupation, magnetization, truncation_error, trotter_error))

#---Plot---#

fig, axes = plt.subplots(len(J_List), 2, figsize=(11, 3.5*len(J_List)), squeeze=False)
colors = plt.cm.viridis(np.linspace(0.1, 0.9, Number_of_Bosonic_Modes))
for row, (J, occupation, magnetization, truncation_error, trotter_error) in enumerate(Scan_Results):
    ax_occ, ax_mag = axes[row]
    for j in range(Number_of_Bosonic_Modes):
        ax_occ.plot(times, occupation[:, j], color=colors[j], lw=1.8, label=f"site {j}")
        ax_mag.plot(times, magnetization[:, j], color=colors[j], lw=1.8)
    ax_occ.set_xlabel("t"); ax_occ.set_ylabel("<n_j>"); ax_occ.set_title(f"J={J}, Boson occupation, trunc err={truncation_error[-1]:.2e}, Trotter err={trotter_error[-1]:.2e}", fontsize=10)
    ax_mag.set_xlabel("t"); ax_mag.set_ylabel("<sz_j>"); ax_mag.set_title("Spin magnetization")
axes[0, 0].legend(fontsize=7, ncol=2)
fig.suptitle(f"TEBD J scan, L={Number_of_Bosonic_Modes} N={Number_of_Fock_States}, max bond={max_bond}")
fig.tight_layout()

with PdfPages(f'Output/TN_Sim_Scan_L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_T{Total_time}.pdf') as pdf:
    pdf.savefig(fig)
print(f'Saved Output/TN_Sim_Scan_L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_T{Total_time}.pdf')
