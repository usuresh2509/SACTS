#!/bin/bash
#SBATCH -J rl_agent        
#SBATCH -A acf-utk0022         
#SBATCH --nodes=1              
#SBATCH --ntasks=1             
#SBATCH --cpus-per-task=16     
#SBATCH --time=10:00:00        
#SBATCH --partition=condo-kvogiatz
#SBATCH --mem=16G              
#SBATCH -e gapjob.e%j        
#SBATCH -o gapjob.o%j        
#SBATCH --qos=condo-kvogiatz

#SBATCH --mail-user=usuresh3@vols.utk.edu
#SBATCH --mail-type=ALL

### GAUSSIAN ###

python -u run.py
