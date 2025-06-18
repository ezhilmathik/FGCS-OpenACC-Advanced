#!/bin/bash -l                            
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --qos=default
#SBATCH --time 00:30:00
#SBATCH --exclusive 

module load env/release/2023.1
module load OpenMPI/4.1.5-NVHPC-23.7-CUDA-12.2.0 # added 17/05/2024
export NVCC_APPEND_FLAGS='-allow-unsupported-compiler' # added 10/04/2024

### for two GPUs ###
nvc -fast -acc=gpu -mp=gpu -gpu=cc80 OMP_STREAM_AN_ISO_2_OpenACC.c Fold3dPMM.c -lgomp -lm -o stream-2
./stream-2
nvc -fast -acc=gpu -mp=gpu -gpu=cc80 P2P_OMP_STREAM_AN_ISO_2_OpenACC.c Fold3dPMM.c -lgomp -lm -o p2p-2
./p2p-2

### for four GPUs ###
nvc -fast -acc=gpu -mp=gpu -gpu=cc80 OMP_STREAM_AN_ISO_2_OpenACC.c Fold3dPMM.c -lgomp -lm -o stream-2
./stream-2
nvc -fast -acc=gpu -mp=gpu -gpu=cc80 P2P_OMP_STREAM_AN_ISO_2_OpenACC.c Fold3dPMM.c -lgomp -lm -o p2p-2
./p2p-2


