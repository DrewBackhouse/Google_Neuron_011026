# Copy to cluster with: rsync -av --exclude '__pycache__' --exclude '.DS_Store' --exclude 'Output' Tensor_Networks_quimb_CREATE/ k2148668@hpc.create.kcl.ac.uk:~/Tensor_Networks_quimb_CREATE/
# Run on cluster with: ./submit.sh --ls "4" --bonds "32 64" --steps "0.1 0.05" --mem 1G --time 6:00:00   (see submit.sh)
# Check all output logs with: grep -h "done in" logs/*.out
# Copy output to mac with: rsync -av k2148668@hpc.create.kcl.ac.uk:~/Tensor_Networks_quimb_CREATE/Output/ Tensor_Networks_quimb_CREATE/Output/
# Then make the plots locally with: python plots.py

import os
import sys
import argparse
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

# Bash script input
parser = argparse.ArgumentParser()
parser.add_argument('--L', type=int, default=Number_of_Bosonic_Modes, help='Number of bosonic modes (sites)')
parser.add_argument('--J', type=float, default=Spin_Interaction_Coefficent, help='Spin-spin interaction coefficient')
parser.add_argument('--max_bond', type=int, default=max_bond)
parser.add_argument('--timestep', type=float, default=Timestep)
parser.add_argument('--tebd_order', type=int, default=tebd_order, choices=[2, 4])
parser.add_argument('--save_every', type=int, default=1, help='Save the .npz every this many measured samples')
args = parser.parse_args()
Number_of_Bosonic_Modes, J, max_bond, Timestep, tebd_order = args.L, args.J, args.max_bond, args.timestep, args.tebd_order

Steps_per_sample = round(0.5/Timestep)

#---Output of Results---#

Output_Dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output', f'L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_T{Total_time}' + ('_PBC' if cyclic else '_OBC')) # Relative to this script, not the working directory
os.makedirs(Output_Dir, exist_ok=True)
Output_Base = os.path.join(Output_Dir, f'Bond{max_bond}_Step{Timestep}_K{Steps_per_sample}_TEBD{tebd_order}_J{J}')
Data_Path = Output_Base + '.npz'
Log_Path = Output_Base + '.out'

Metadata = dict(J=J, L=Number_of_Bosonic_Modes, N=Number_of_Fock_States, D=Displacement_Coefficent, Spin_Boson_Interaction_Coefficent=Spin_Boson_Interaction_Coefficent,
                Total_time=Total_time, Timestep=Timestep, Steps_per_sample=Steps_per_sample, max_bond=max_bond, tebd_order=tebd_order, cyclic=cyclic)

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
    print(f"Running L={Number_of_Bosonic_Modes}, J={J}: max_bond={max_bond}, Timestep={Timestep}, tebd_order={tebd_order}, steps_per_sample={Steps_per_sample}, cyclic={cyclic}, saving to {Data_Path} every {args.save_every} samples", flush=True)
    times, occupation, magnetization, truncation_error, trotter_error, tn_time = TNSim(Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, J, Spin_Boson_Interaction_Coefficent, Total_time, max_bond, Timestep, tebd_order, Steps_per_sample, cyclic,
                                                                                      data_path=Data_Path, metadata=Metadata, save_every=args.save_every)
    print(f"J={J}: TEBD (max_bond={max_bond}) done in {tn_time:.2f}s, final truncation error={truncation_error[-1]:.2e}, Trotter error={trotter_error[-1]:.2e}", flush=True)
    print(f'Saved {Data_Path}', flush=True)
