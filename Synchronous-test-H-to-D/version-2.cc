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

  
  // Initialize vectors
  for (int i = 0; i < N; i++) {
    h_A[i] = 1.0;
    h_B[i] = 1.0;
  }

  acc_set_device_num(0, acc_device_nvidia);
  double *d_A = (double*) acc_malloc(bytes);
  acc_set_device_num(1, acc_device_nvidia);
  double *d_B = (double*) acc_malloc(bytes);
  
  omp_set_num_threads(2);
  
  double start = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp sections
    { 
#pragma omp section
      {
	acc_set_device_num(0, acc_device_nvidia);
	acc_memcpy_to_device(d_A, h_A, bytes);
      }
#pragma omp section
      {      
	acc_set_device_num(1, acc_device_nvidia);
	acc_memcpy_to_device(d_B, h_B, bytes);
      }
    }
  }
  
  double end = omp_get_wtime();
  double milliseconds = (end - start) * 1000.0;  

  // Convert to seconds and compute bandwidth (2 copies)
  double seconds = milliseconds / 1000.0;
  double total_bytes = 2 * bytes; // two transfers: h_A → d_A and h_B → d_B
  double bandwidthGBps = total_bytes / (seconds * 1e9);
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Elapsed time: " << seconds << " seconds\n";
  std::cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  std::cout << "PCIe Bandwidth: " << bandwidthGBps << " GB/s\n";
  
  // Free device memory
  acc_set_device_num(0, acc_device_nvidia);
  acc_free(d_A);
  acc_set_device_num(1, acc_device_nvidia);
  acc_free(d_B);

  // Free host memory
  free(h_A);
  free(h_B);
  
  return 0;
}
