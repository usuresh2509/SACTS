#!/bin/bash
#SBATCH -J berny
#SBATCH -A {your_account}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --partition={your_partition}
#SBATCH --qos={your_qos}

python -u run.py
