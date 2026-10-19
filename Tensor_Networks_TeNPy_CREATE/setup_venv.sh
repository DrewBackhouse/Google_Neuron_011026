#!/bin/bash
# Builds TeNPyvenv in this folder from requirements.txt. Run once on the CREATE login node (submit.sh calls it automatically if TeNPyvenv is missing)
# The venv has to be built on the cluster: one made on a Mac holds macOS binaries and links to the Mac's Python, so it won't run on the Linux nodes
# To rebuild from scratch: rm -rf TeNPyvenv && ./setup_venv.sh

set -e
cd "$(dirname "$0")"

module load python/3.11.6-gcc-13.2.0 # Must match the module loaded in Bash_Script.sh
python -m venv TeNPyvenv
TeNPyvenv/bin/python -m pip install --upgrade pip
TeNPyvenv/bin/python -m pip install -r requirements.txt
TeNPyvenv/bin/python -c "import tenpy, numpy, scipy; print(f'TeNPyvenv ready: tenpy {tenpy.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}')"
