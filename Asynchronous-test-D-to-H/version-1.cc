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

  // Initialize vectors
  for (int i = 0; i < N; i++) {
    h_A[i] = 1.0;
  }

  acc_set_device_num(0, acc_device_nvidia);
  double *d_A = (double*) acc_malloc(bytes);
  acc_memcpy_to_device(d_A, h_A, bytes);
  
  // Measure execution time
  double start = omp_get_wtime();
  acc_set_device_num(0, acc_device_nvidia);
  acc_memcpy_from_device_async(h_A, d_A, bytes,0);
  double end = omp_get_wtime();
  double milliseconds = (end - start) * 1000.0;  

  // Convert to seconds and compute bandwidth (2 copies)
  double seconds = milliseconds / 1000.0;
  double total_bytes = bytes; // two transfers: h_A → d_A and h_B → d_B
  double bandwidthGBps = total_bytes / (seconds * 1e9);
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Elapsed time: " << seconds << " seconds\n";
  std::cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  std::cout << "PCIe Bandwidth: " << bandwidthGBps << " GB/s\n";
  
  // Free device memory
  acc_set_device_num(0, acc_device_nvidia);
  acc_free(d_A);

  // Free host memory
  free(h_A);
  
  return 0;
}
