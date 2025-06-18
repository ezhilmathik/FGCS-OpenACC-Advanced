#!/bin/bash -l                            
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time 00:30:00                                              
#SBATCH --exclusive 

module load env/release/2023.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0 # added 17/05/2024
export NVCC_APPEND_FLAGS='-allow-unsupported-compiler' # added 10/04/2024

nvcc -O3 -Xcompiler -fopenmp -arch=sm_80 -o version-1-cuda version-1.cu
nvcc -O3 -Xcompiler -fopenmp -arch=sm_80 -o version-2-cuda version-2.cu
nvcc -O3 -Xcompiler -fopenmp -arch=sm_80 -o version-4-cuda version-4.cu
nvc++ -fast -mp=gpu -gpu=cc80 -Minfo=accel -lcudart -o version-1-acc version-1.cc
nvc++ -fast -mp=gpu -gpu=cc80 -Minfo=accel -lcudart -o version-2-acc version-2.cc
nvc++ -fast -mp=gpu -gpu=cc80 -Minfo=accel -lcudart -o version-4-acc version-4.cc

Ns=(940000000 1740000000 2550000000 3350000000 4160000000 4960000000)

for N in "${Ns[@]}"
do
    echo " Begin 1 GPU test"
    echo "Running CUDA Together test for N = $N"
    ./version-1-cuda "$N"
    sleep 15s
    echo "Running OpenACC Together test for N = $N"
    ./version-1-acc "$N"
    sleep 15s
    echo ""
    echo " Begin 2 GPU test"
    echo "Running CUDA Together test for N = $N"
    ./version-2-cuda "$N"
    sleep 15s
    echo "Running OpenACC Together test for N = $N"
    ./version-2-acc "$N"
    sleep 15s
    echo ""
    echo " Begin 4 GPU test"
    echo "Running CUDA Seperately test for N = $N"
    ./version-4-cuda "$N"
    sleep 15s
    echo "Running OpenACC Seperately test for N = $N"
    ./version-4-acc "$N"
    sleep 15s
    echo "==========================="
done
