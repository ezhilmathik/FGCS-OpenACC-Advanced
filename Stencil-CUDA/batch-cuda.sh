#!/bin/bash -l                            
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time 00:10:00                                              
#SBATCH --exclusive 

module load env/staging/2023.1 # added 22/04/2024
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-11.7.0 # added 10/04/2024
#export NVCC_APPEND_FLAGS='-allow-unsupported-compiler' # added 10/04/2024

#module load env/release/2023.1
#module load CUDA/12.2.0

nvcc P2P_OMP_AN_ISO_4_GPU.cu Fold3dPMM.c -arch compute_80 -lm -lgomp -Xcompiler -fopenmp

./a.out 

