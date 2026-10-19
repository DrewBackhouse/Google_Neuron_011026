# Copy to cluster with: rsync -av --exclude '__pycache__' --exclude '.DS_Store' --exclude 'Output' --exclude 'TeNPyvenv' --exclude 'logs' Tensor_Networks_TeNPy_CREATE/ k2148668@hpc.create.kcl.ac.uk:~/Tensor_Networks_TeNPy_CREATE/
# Run on cluster with: ./submit.sh --ls "4" --bonds "32 64" --steps "0.1 0.05" --mem 1G --time 6:00:00   (see submit.sh, the first run also builds TeNPyvenv with setup_venv.sh)
# Check all output logs with: grep -h "done in" logs/*.out
# Copy output to mac with: rsync -av k2148668@hpc.create.kcl.ac.uk:~/Tensor_Networks_TeNPy_CREATE/Output/ Tensor_Networks_TeNPy_CREATE/Output/
# TEBD (TeNPy's QRBasedTEBDEngine) is used for open chains and L <= 2, periodic chains with L > 2 use two-site TDVP as TEBD needs nearest neighbour bonds
# Then make the plots locally with: python plots.py

import os
import sys
import argparse
import numpy as np
from Functions import TNSim

#---Model Parameters---#

Number_of_Fock_States = 8
Number_of_Bosonic_Modes = 1
Displacement_Coefficent = 1
Spin_Interaction_Coefficent = 0.1 # J
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip
cyclic = True

#---Simulation Parameters---#

# Defults
Total_time = 100
Timestep = 0.1
max_bond = 32
tebd_order = 2
cbe_expand = 0.1 # QR-TEBD bond expansion rate, see Functions.TNSim
cbe_expand_0 = None # QR-TEBD expansion rate at bond dimension 1, None uses cbe_expand throughout

# Bash script input
parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, default=Number_of_Bosonic_Modes, help='Number of bosonic modes (sites)')
parser.add_argument('--J', type=float, default=Spin_Interaction_Coefficent, help='Spin-spin interaction coefficient')
parser.add_argument('--max_bond', type=int, default=max_bond)
parser.add_argument('--timestep', type=float, default=Timestep)
parser.add_argument('--tebd_order', type=int, default=tebd_order, choices=[2, 4])
parser.add_argument('--cbe_expand', type=float, default=cbe_expand, help='QR-TEBD bond expansion rate per step')
parser.add_argument('--cbe_expand_0', type=float, default=cbe_expand_0, help='QR-TEBD expansion rate at bond dimension 1 (left out: cbe_expand throughout)')
parser.add_argument('--save_every', type=int, default=1, help='Save the .npz every this many measured samples')
parser.add_argument('--bc', choices=['pbc', 'obc'], default=('pbc' if cyclic else 'obc'), help='Periodic or open boundary conditions')
args = parser.parse_args()
Number_of_Bosonic_Modes, J, max_bond, Timestep, tebd_order, cbe_expand, cbe_expand_0 = args.L, args.J, args.max_bond, args.timestep, args.tebd_order, args.cbe_expand, args.cbe_expand_0
cyclic = args.bc == 'pbc'

Steps_per_sample = round(0.5/Timestep)
Algorithm = 'TDVP' if (cyclic and Number_of_Bosonic_Modes > 2) else 'QR-TEBD' # Must match the choice in TNSim
if Algorithm == 'TDVP':
    cbe_expand = cbe_expand_0 = None # Not used by TDVP
CBE_Tag = '' if cbe_expand is None else f'_CBE{cbe_expand}' + ('' if cbe_expand_0 is None else f'-{cbe_expand_0}') # Same format as cbe_tag in plots.py

#---Output of Results---#

Output_Dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output', f'L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_T{Total_time}' + ('_PBC' if cyclic else '_OBC')) # Relative to this script, not the working directory
os.makedirs(Output_Dir, exist_ok=True)
Output_Base = os.path.join(Output_Dir, f'Bond{max_bond}_Step{Timestep}_K{Steps_per_sample}_TEBD{tebd_order}{CBE_Tag}_J{J}')
Data_Path = Output_Base + '.npz'
Log_Path = Output_Base + '.out'

Metadata = dict(J=J, L=Number_of_Bosonic_Modes, N=Number_of_Fock_States, D=Displacement_Coefficent, Spin_Boson_Interaction_Coefficent=Spin_Boson_Interaction_Coefficent,
                Total_time=Total_time, Timestep=Timestep, Steps_per_sample=Steps_per_sample, max_bond=max_bond, tebd_order=tebd_order, cyclic=cyclic, algorithm=Algorithm,
                cbe_expand=np.nan if cbe_expand is None else cbe_expand, cbe_expand_0=np.nan if cbe_expand_0 is None else cbe_expand_0) # NaN as np.savez can't store None

class Tee:
    # Copies everything printed to both the terminal (or SLURM log) and the log file
    def __init__(self, *streams):
        self.streams = streams
    def write(self, text):
        for stream in self.streams:
            stream.write(text)
    def flush(self):
        for stream in self.streams:
            stream.flush()

#---TN Simulation---#

if __name__ == '__main__':
    Log_File = open(Log_Path, 'w')
    sys.stdout = Tee(sys.__stdout__, Log_File)
    sys.stderr = Tee(sys.__stderr__, Log_File) # Tracebacks also end up in the log
    print(f"Running {Algorithm} L={Number_of_Bosonic_Modes}, J={J}: max_bond={max_bond}, Timestep={Timestep}, tebd_order={tebd_order}, cbe_expand={cbe_expand}, cbe_expand_0={cbe_expand_0}, steps_per_sample={Steps_per_sample}, cyclic={cyclic}, saving to {Data_Path} every {args.save_every} samples", flush=True)
    times, occupation, magnetization, truncation_error, trotter_error, tn_time = TNSim(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, J, Spin_Boson_Interaction_Coefficent, Total_time, max_bond, Timestep, tebd_order, Steps_per_sample, cyclic,
                                                                                      data_path=Data_Path, cbe_expand=(0.1 if cbe_expand is None else cbe_expand), cbe_expand_0=cbe_expand_0, metadata=Metadata, save_every=args.save_every)
    print(f"J={J}: {Algorithm} (max_bond={max_bond}) done in {tn_time:.2f}s, final truncation error={truncation_error[-1]:.2e}, Trotter error={trotter_error[-1]:.2e}", flush=True)
    print(f'Saved {Data_Path}', flush=True)
