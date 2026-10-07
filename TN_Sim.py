import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from Functions import TNHamiltonian, TNSim

#---Model Parameters---#

Number_of_Fock_States = 10
Number_of_Bosonic_Modes = 5
Displacement_Coefficent = 1
Spin_Interaction_Coefficent = 1
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip

#---Simulation Parameters---#

Total_time = 100
Timepoints = 200
max_bond = None
Timestep = 0.02
tebd_order = 4

#---TN Sim---#

times = np.linspace(0, Total_time, Timepoints)
t0 = time.time()
occupation, magnetization, truncation_error, trotter_error = TNSim(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, Spin_Interaction_Coefficent, Spin_Boson_Interaction_Coefficent, times, max_bond, Timestep, tebd_order, progbar=False)
tn_time = time.time() - t0
print(f"TEBD (max_bond={max_bond}) done in {tn_time:.2f}s, final truncation error={truncation_error[-1]:.2e}, Trotter error={trotter_error[-1]:.2e}")

fig, (ax_occ, ax_mag) = plt.subplots(1, 2, figsize=(11, 4.5))
colors = plt.cm.viridis(np.linspace(0.1, 0.9, Number_of_Bosonic_Modes))
for j in range(Number_of_Bosonic_Modes):
    ax_occ.plot(times, occupation[:, j], color=colors[j], lw=1.8, label=f"site {j}")
    ax_mag.plot(times, magnetization[:, j], color=colors[j], lw=1.8)
ax_occ.set_xlabel("t"); ax_occ.set_ylabel("<n_j>"); ax_occ.set_title("Boson occupation")
ax_mag.set_xlabel("t"); ax_mag.set_ylabel("<sz_j>"); ax_mag.set_title("Spin magnetization")
ax_occ.legend(fontsize=7, ncol=2)
fig.suptitle(f"TEBD only, L={Number_of_Bosonic_Modes} N={Number_of_Fock_States}, max bond={max_bond}, truncation error={truncation_error[-1]:.2e}, Trotter error={trotter_error[-1]:.2e}")
fig.tight_layout()

with PdfPages(f'Output/TN_Sim_L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_T{Total_time}.pdf') as pdf:
    pdf.savefig(fig)
print(f'Saved Output/TN_Sim_L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_T{Total_time}.pdf')
