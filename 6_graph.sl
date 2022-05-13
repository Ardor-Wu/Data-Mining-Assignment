#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --constraint=haswell

srun python ships.py 6 --algorithm graph
