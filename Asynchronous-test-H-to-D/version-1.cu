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

  for (size_t i = 0; i < N; ++i) {
    h_A[i] = 1.0;

  }

  // Allocate device memory
  double *d_A = nullptr;
  cudaSetDevice(0);
  checkCudaError(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");

  int numberofgpus = 1;      
  cudaStream_t halo_stream[numberofgpus];
    
  // create a streams 
  for (int i = 0; i < numberofgpus; ++i) 
    {
      cudaSetDevice(i);
      cudaStreamCreate(&halo_stream[i]);
    }

  // Start timing
  cudaEvent_t start, stop;
  checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
  checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");
  checkCudaError(cudaEventRecord(start), "cudaEventRecord start");

  // Copy data to both GPUs
  cudaSetDevice(0);
  checkCudaError(cudaMemcpyAsync(d_A, h_A, bytes, cudaMemcpyHostToDevice, halo_stream[0]), "cudaMemcpy d_A");
  cudaSetDevice(0);
  cudaStreamSynchronize(halo_stream[0]);
  checkCudaError(cudaEventRecord(stop), "cudaEventRecord stop");
  checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");
    
  float milliseconds = 0;
  checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    
  // Convert to seconds and compute bandwidth (2 copies)
  double seconds = milliseconds / 1000.0;
  double total_bytes = bytes; // two transfers: h_A → d_A and h_B → d_B
  double bandwidthGBps = total_bytes / (seconds * 1e9);
    
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Elapsed time: " << seconds << " seconds\n";
  std::cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  std::cout << "PCIe Bandwidth: " << bandwidthGBps << " GB/s\n";
    
  // Clean up
  cudaSetDevice(0);
  checkCudaError(cudaFree(d_A), "cudaFree d_A");    
  delete[] h_A;
  checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
  checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");

  return EXIT_SUCCESS;
}
