#!/bin/bash
#SBATCH -J fermi_level
#SBATCH -A acf-utk0022
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --partition=short
#SBATCH --qos=short

export OMP_NUM_THREADS=16
export OMP_STACKSIZE=2G
ulimit -s unlimited
python berny.py