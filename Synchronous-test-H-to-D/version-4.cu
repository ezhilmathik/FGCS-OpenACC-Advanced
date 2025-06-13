//-*-c++-*-
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <omp.h>

void checkCudaError(cudaError_t result, const char *function) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA error in " << function << ": " << cudaGetErrorString(result) << std::endl;
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <vector_size>" << std::endl;
    return EXIT_FAILURE;
  }
  
  size_t N = std::stoull(argv[1]);
  size_t bytes = N * sizeof(double);
  
  // Allocate and initialize host memory
  double *h_A = new double[N];
  double *h_B = new double[N];
  double *h_C = new double[N];
  double *h_D = new double[N];
  for (size_t i = 0; i < N; ++i)
    {
      h_A[i] = 1.0;
      h_B[i] = 1.0;
      h_C[i] = 1.0;
      h_D[i] = 1.0;	
    }
  
  // Allocate device memory
  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;
  cudaSetDevice(0);
  checkCudaError(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
  cudaSetDevice(1);
  checkCudaError(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
  cudaSetDevice(2);
  checkCudaError(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");
  cudaSetDevice(3);
  checkCudaError(cudaMalloc(&d_D, bytes), "cudaMalloc d_D");
  
  omp_set_num_threads(4);
  double start = omp_get_wtime();    
#pragma omp parallel
  {
#pragma omp sections
    { 
#pragma omp section
      {
	cudaSetDevice(0);
	checkCudaError(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_A");
      }
#pragma omp section
      {	  
	cudaSetDevice(1);
	checkCudaError(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_B");    
      }
#pragma omp section
      {	  
	cudaSetDevice(2);
	checkCudaError(cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_C");    
      }
#pragma omp section
      {	  
	cudaSetDevice(3);
	checkCudaError(cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_D");    
      }
    }
  }
  
  double end = omp_get_wtime();
  double milliseconds = (end - start) * 1000.0;
  
  double seconds = milliseconds / 1000.0;
  double total_bytes = 4 * bytes; // two transfers: h_A → d_A and h_B → d_B
  double bandwidthGBps = total_bytes / (seconds * 1e9);
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Elapsed time: " << seconds << " seconds\n";
  std::cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  std::cout << "PCIe Bandwidth: " << bandwidthGBps << " GB/s\n";
  
  // Clean up
  cudaSetDevice(0);
  checkCudaError(cudaFree(d_A), "cudaFree d_A");
  cudaSetDevice(1);
  checkCudaError(cudaFree(d_B), "cudaFree d_B");
  cudaSetDevice(2);
  checkCudaError(cudaFree(d_C), "cudaFree d_C");
  cudaSetDevice(3);
  checkCudaError(cudaFree(d_D), "cudaFree d_D");
  
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_D;
  
  return EXIT_SUCCESS;
}
