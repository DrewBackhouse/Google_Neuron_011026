# Copy to cluster with: rsync -av --exclude '__pycache__' --exclude '.DS_Store' --exclude 'Output' Tensor_Networks_quimb/ k2148668@hpc.create.kcl.ac.uk:~/Tensor_Networks_quimb/
# Run on cluster with: mkdir -p logs && sbatch Bash_Script.sh
# Copy output to mac with: scp -r k2148668@hpc.create.kcl.ac.uk:~/Tensor_Networks_quimb/Output ~/Downloads/

import os
import sys
#import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import numpy as np
#from matplotlib.backends.backend_pdf import PdfPages

from Functions import save_data, save_results, run_J

#---Model Parameters---#

Number_of_Fock_States = 8
Number_of_Bosonic_Modes = 3
Displacement_Coefficent = 1
J_List = np.round(np.arange(0.1, 0.1 + 1e-9, 0.1), 6)
Spin_Boson_Interaction_Coefficent = Displacement_Coefficent * (Number_of_Fock_States -0.5)**0.5 # Value for a full spin flip

#---Simulation Parameters---#

# Defults
Total_time = 5
Timestep = 0.1
max_bond = 32
tebd_order = 2

# Bash script input
parser = argparse.ArgumentParser()
parser.add_argument('--max_bond', type=int, default=max_bond)
parser.add_argument('--timestep', type=float, default=Timestep)
parser.add_argument('--tebd_order', type=int, default=tebd_order, choices=[2, 4])
parser.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count())), help='Number of J values run in parallel')
args = parser.parse_args()
max_bond, Timestep, tebd_order = args.max_bond, args.timestep, args.tebd_order

Steps_per_sample = round(0.5/Timestep)

#---Output of Results---#

Output_Dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output', f'L{Number_of_Bosonic_Modes}_N{Number_of_Fock_States}_J{J_List[0]}-{J_List[-1]}_T{Total_time}') # Relative to this script, not the working directory
os.makedirs(Output_Dir, exist_ok=True)
Output_Base = os.path.join(Output_Dir, f'Bond{max_bond}_Step{Timestep}_TEBD{tebd_order}')
Output_Path = Output_Base + '.pdf'
Data_Path = Output_Base + '.npz'
Log_Path = Output_Base + '.out'

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

#---Scan over J (in parallel)---#

if __name__ == '__main__':
    Log_File = open(Log_Path, 'w')
    sys.stdout = Tee(sys.__stdout__, Log_File)
    sys.stderr = Tee(sys.__stderr__, Log_File) # Tracebacks also end up in the log
    workers = max(1, min(args.workers, len(J_List)))
    print(f"Running {len(J_List)} J values on {workers} workers: max_bond={max_bond}, Timestep={Timestep}, tebd_order={tebd_order}, steps_per_sample={Steps_per_sample}", flush=True)
    Scan_Results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_J, J, Number_of_Bosonic_Modes, Number_of_Fock_States, Displacement_Coefficent, Spin_Boson_Interaction_Coefficent, Total_time, max_bond, Timestep, tebd_order, Steps_per_sample) for J in J_List]
        for future in as_completed(futures):
            J, times, occupation, magnetization, truncation_error, trotter_error, tn_time = future.result()
            print(f"J={J}: TEBD (max_bond={max_bond}) done in {tn_time:.2f}s, final truncation error={truncation_error[-1]:.2e}, Trotter error={trotter_error[-1]:.2e}", flush=True)
            Scan_Results.append((J, times, occupation, magnetization, truncation_error, trotter_error, tn_time))
            Scan_Results.sort(key=lambda result: result[0]) # Runs finish out of order, keep plots ordered by J
            save_results(Scan_Results, Output_Path, Number_of_Bosonic_Modes, Number_of_Fock_States, Total_time, Timestep, Steps_per_sample, max_bond, tebd_order)
            save_data(Scan_Results, Data_Path, Number_of_Bosonic_Modes, Number_of_Fock_States, Total_time, Timestep, Steps_per_sample, max_bond, tebd_order)
            print(f'Saved {Output_Path} and {Data_Path}', flush=True)
