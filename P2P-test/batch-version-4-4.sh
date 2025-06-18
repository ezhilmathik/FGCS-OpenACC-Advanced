#!/bin/bash -l                            
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time 00:30:00                                              
#SBATCH --exclusive 

module load env/release/2023.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0 # added 17/05/2024
export NVCC_APPEND_FLAGS='-allow-unsupported-compiler' # added 10/04/2024

Ns=(58750000 108750000 159375000 209375000 260000000 310000000)

nvcc -O3 -Xcompiler -fopenmp -arch=sm_80 -o version-4-cuda version-4.cu
nvc++ -fast -mp=gpu -gpu=cc80 -Minfo=accel -lcudart -o version-4-acc version-4.cc

for N in "${Ns[@]}"
do
    echo " Begin 4 GPU test"
    echo "Running CUDA Seperately test for N = $N"
    ./version-4-cuda "$N"
    sleep 15s
    echo "Running OpenACC Seperately test for N = $N"
    ./version-4-acc "$N"
    sleep 15s
    echo "==========================="
done
