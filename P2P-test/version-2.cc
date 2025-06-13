#include <iostream>
#include <iomanip>
#include <cmath>
#include <openacc.h>
#include <omp.h>
#include <fstream>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <vector_size>" << std::endl;
    return EXIT_FAILURE;
  }
  
  size_t N = std::stoull(argv[1]);
  size_t bytes = N * sizeof(double);
  
  // Allocate host memory
  double *h_A = (double*) malloc(bytes);
  double *h_B = (double*) malloc(bytes);
  double *h_C = (double*) malloc(bytes);
  double *h_D = (double*) malloc(bytes);
    
  if (h_A == nullptr || h_B == nullptr) {
    std::cerr << "Error: Host memory allocation failed." << std::endl;
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);    
    return 1;
  }
  
  // Initialize vectors
  for (int i = 0; i < N; i++) {
    h_A[i] = 1.0;
    h_B[i] = 2.0;
    h_C[i] = 3.0;
    h_D[i] = 4.0;    
  }

  // Device 0
  acc_set_device_num(0, acc_device_nvidia);
  double* d_A = (double*) acc_malloc(bytes);
  double* d_D = (double*) acc_malloc(bytes);
  acc_memcpy_to_device(d_A, h_A, bytes);
  acc_memcpy_to_device(d_D, h_D, bytes);
  
  // Device 1
  acc_set_device_num(1, acc_device_nvidia);
  double* d_B = (double*) acc_malloc(bytes);
  double* d_C = (double*) acc_malloc(bytes);
  acc_memcpy_to_device(d_B, h_B, bytes);
  acc_memcpy_to_device(d_C, h_C, bytes);
  
  
  omp_set_num_threads(2);
  double start = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      {
	acc_set_device_num(0, acc_device_nvidia);
	acc_memcpy_device_async(d_C, d_A, bytes, 0);
	acc_wait_all();
      }
#pragma omp section
      {	
	acc_set_device_num(1, acc_device_nvidia);
	acc_memcpy_device_async(d_D, d_B, bytes, 1);
	acc_wait_all();
      }
    }
  }

  double end = omp_get_wtime();
  double milliseconds = (end - start) * 1000.0;  
  
  // Convert to seconds and compute bandwidth (2 copies)
  double seconds = milliseconds / 1000.0;
  double total_bytes = 2*bytes; // two transfers: 
  double bandwidthGBps = total_bytes / (seconds * 1e9);
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Elapsed time: " << seconds << " seconds\n";
  std::cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  std::cout << "PCIe Bandwidth: " << bandwidthGBps << " GB/s\n";

  
  // Free device memory
  acc_set_device_num(0, acc_device_nvidia);
  acc_free(d_A);
  acc_free(d_D);
  acc_set_device_num(1, acc_device_nvidia);
  acc_free(d_B);
  acc_free(d_C);

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_D);
  
  return 0;
}
