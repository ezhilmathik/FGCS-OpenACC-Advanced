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
  
  // Allocate and initialize host memory
  double *h_A1 = (double*) malloc(bytes);
  double *h_B1 = (double*) malloc(bytes);
  double *h_C1 = (double*) malloc(bytes);
  double *h_D1 = (double*) malloc(bytes);

  double *h_A2 = (double*) malloc(bytes);
  double *h_B2 = (double*) malloc(bytes);
  double *h_C2 = (double*) malloc(bytes);
  double *h_D2 = (double*) malloc(bytes);
  
  double *h_A3 = (double*) malloc(bytes);
  double *h_B3 = (double*) malloc(bytes);
  double *h_C3 = (double*) malloc(bytes);
  double *h_D3 = (double*) malloc(bytes);

  double *h_A4 = (double*) malloc(bytes);
  double *h_B4 = (double*) malloc(bytes);
  double *h_C4 = (double*) malloc(bytes);
  double *h_D4 = (double*) malloc(bytes);

  for (size_t i = 0; i < N; ++i) {
    h_A1[i] = 1.0;
    h_B1[i] = 1.0;
    h_C1[i] = 1.0;
    h_D1[i] = 1.0;	

    h_A2[i] = 2.0;
    h_B2[i] = 2.0;
    h_C2[i] = 2.0;
    h_D2[i] = 2.0;	

    h_A3[i] = 3.0;
    h_B3[i] = 3.0;
    h_C3[i] = 3.0;
    h_D3[i] = 3.0;	
      
    h_A4[i] = 4.0;
    h_B4[i] = 4.0;
    h_C4[i] = 4.0;
    h_D4[i] = 4.0;	
  }
    
    
  // Allocate device memory
  //  double *d_A1 = nullptr, *d_B1 = nullptr, *d_C1 = nullptr, *d_D1 = nullptr;
  //  double *d_A2 = nullptr, *d_B2 = nullptr, *d_C2 = nullptr, *d_D2 = nullptr;
  //  double *d_A3 = nullptr, *d_B3 = nullptr, *d_C3 = nullptr, *d_D3 = nullptr;
  //  double *d_A4 = nullptr, *d_B4 = nullptr, *d_C4 = nullptr, *d_D4 = nullptr;
    
  acc_set_device_num(0, acc_device_nvidia);
  double* d_A1 = (double*) acc_malloc(bytes);
  double* d_B1 = (double*) acc_malloc(bytes);
  double* d_C1 = (double*) acc_malloc(bytes);
  double* d_D1 = (double*) acc_malloc(bytes);
  double* d_A11 = (double*) acc_malloc(bytes);
  double* d_B11 = (double*) acc_malloc(bytes);
  double* d_C11 = (double*) acc_malloc(bytes);
  double* d_D11 = (double*) acc_malloc(bytes);

  acc_memcpy_to_device(d_A1, h_A1, bytes);
  acc_memcpy_to_device(d_B1, h_B1, bytes);
  acc_memcpy_to_device(d_C1, h_C1, bytes);
  acc_memcpy_to_device(d_D1, h_D1, bytes);

  acc_memcpy_to_device(d_A11, h_A1, bytes);
  acc_memcpy_to_device(d_B11, h_B1, bytes);
  acc_memcpy_to_device(d_C11, h_C1, bytes);
  acc_memcpy_to_device(d_D11, h_D1, bytes);

  acc_set_device_num(1, acc_device_nvidia);
  double* d_A2 = (double*) acc_malloc(bytes);
  double* d_B2 = (double*) acc_malloc(bytes);
  double* d_C2 = (double*) acc_malloc(bytes);
  double* d_D2 = (double*) acc_malloc(bytes);
  double* d_A22 = (double*) acc_malloc(bytes);
  double* d_B22 = (double*) acc_malloc(bytes);
  double* d_C22 = (double*) acc_malloc(bytes);
  double* d_D22 = (double*) acc_malloc(bytes);

  acc_memcpy_to_device(d_A2, h_A2, bytes);
  acc_memcpy_to_device(d_B2, h_B2, bytes);
  acc_memcpy_to_device(d_C2, h_C2, bytes);
  acc_memcpy_to_device(d_D2, h_D2, bytes);

  acc_memcpy_to_device(d_A22, h_A2, bytes);
  acc_memcpy_to_device(d_B22, h_B2, bytes);
  acc_memcpy_to_device(d_C22, h_C2, bytes);
  acc_memcpy_to_device(d_D22, h_D2, bytes);

  acc_set_device_num(2, acc_device_nvidia);
  double* d_A3 = (double*) acc_malloc(bytes);
  double* d_B3 = (double*) acc_malloc(bytes);
  double* d_C3 = (double*) acc_malloc(bytes);
  double* d_D3 = (double*) acc_malloc(bytes);
  double* d_A33 = (double*) acc_malloc(bytes);
  double* d_B33 = (double*) acc_malloc(bytes);
  double* d_C33 = (double*) acc_malloc(bytes);
  double* d_D33 = (double*) acc_malloc(bytes);

  acc_memcpy_to_device(d_A3, h_A3, bytes);
  acc_memcpy_to_device(d_B3, h_B3, bytes);
  acc_memcpy_to_device(d_C3, h_C3, bytes);
  acc_memcpy_to_device(d_D3, h_D3, bytes);

  acc_memcpy_to_device(d_A33, h_A3, bytes);
  acc_memcpy_to_device(d_B33, h_B3, bytes);
  acc_memcpy_to_device(d_C33, h_C3, bytes);
  acc_memcpy_to_device(d_D33, h_D3, bytes);

  acc_set_device_num(3, acc_device_nvidia);
  double* d_A4 = (double*) acc_malloc(bytes);
  double* d_B4 = (double*) acc_malloc(bytes);
  double* d_C4 = (double*) acc_malloc(bytes);
  double* d_D4 = (double*) acc_malloc(bytes);
  double* d_A44 = (double*) acc_malloc(bytes);
  double* d_B44 = (double*) acc_malloc(bytes);
  double* d_C44 = (double*) acc_malloc(bytes);
  double* d_D44 = (double*) acc_malloc(bytes);

  acc_memcpy_to_device(d_A4, h_A4, bytes);
  acc_memcpy_to_device(d_B4, h_B4, bytes);
  acc_memcpy_to_device(d_C4, h_C4, bytes);
  acc_memcpy_to_device(d_D4, h_D4, bytes);

  acc_memcpy_to_device(d_A44, h_A4, bytes);
  acc_memcpy_to_device(d_B44, h_B4, bytes);
  acc_memcpy_to_device(d_C44, h_C4, bytes);
  acc_memcpy_to_device(d_D44, h_D4, bytes);

  

  omp_set_num_threads(4);
  double start = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp sections
    {
#pragma omp section
      {
	acc_set_device_num(0, acc_device_nvidia);
	acc_memcpy_device_async(d_A2, d_B11, bytes, 0);
	acc_memcpy_device_async(d_A3, d_C11, bytes, 1);
	acc_memcpy_device_async(d_A4, d_D11, bytes, 2);
	acc_wait_all();
      }
#pragma omp section
      {	
	acc_set_device_num(1, acc_device_nvidia);
	acc_memcpy_device_async(d_B2, d_A22, bytes, 0);
	acc_memcpy_device_async(d_B3, d_C22, bytes, 1);
	acc_memcpy_device_async(d_B4, d_D22, bytes, 2);	
	acc_wait_all();
      }
#pragma omp section
      {	
	acc_set_device_num(2, acc_device_nvidia);	
	acc_memcpy_device_async(d_C1, d_A33, bytes, 0);
	acc_memcpy_device_async(d_C2, d_B33, bytes, 1);
	acc_memcpy_device_async(d_C4, d_D33, bytes, 2);	
	acc_wait_all();
      }
#pragma omp section
      {	
	acc_set_device_num(3, acc_device_nvidia);	
	acc_memcpy_device_async(d_D1, d_A44, bytes, 0);
	acc_memcpy_device_async(d_D2, d_B44, bytes, 1);
	acc_memcpy_device_async(d_D3, d_C44, bytes, 2);	
	acc_wait_all();
      }
    }
  }

  double end = omp_get_wtime();
  double milliseconds = (end - start) * 1000.0;  
  
  // Convert to seconds and compute bandwidth (2 copies)
  double seconds = milliseconds / 1000.0;
  double total_bytes = 12*bytes; // two transfers: 
  double bandwidthGBps = total_bytes / (seconds * 1e9);
  
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Elapsed time: " << seconds << " seconds\n";
  std::cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
  std::cout << "PCIe Bandwidth: " << bandwidthGBps << " GB/s\n";

  // Free device memory
  acc_set_device_num(0, acc_device_nvidia);
  acc_free(d_A1);
  acc_free(d_B1);
  acc_free(d_C1);
  acc_free(d_D1);
  acc_free(d_A11);
  acc_free(d_B11);
  acc_free(d_C11);
  acc_free(d_D11);

  acc_set_device_num(1, acc_device_nvidia);
  acc_free(d_A2);
  acc_free(d_B2);
  acc_free(d_C2);
  acc_free(d_D2);
  acc_free(d_A22);
  acc_free(d_B22);
  acc_free(d_C22);
  acc_free(d_D22);

  acc_set_device_num(2, acc_device_nvidia);
  acc_free(d_A3);
  acc_free(d_B3);
  acc_free(d_C3);
  acc_free(d_D3);
  acc_free(d_A33);
  acc_free(d_B33);
  acc_free(d_C33);
  acc_free(d_D33);

  acc_set_device_num(3, acc_device_nvidia);
  acc_free(d_A4);
  acc_free(d_B4);
  acc_free(d_C4);
  acc_free(d_D4);
  acc_free(d_A44);
  acc_free(d_B44);
  acc_free(d_C44);
  acc_free(d_D44);
  
  // Free host memory
    delete[] h_A1;
    delete[] h_B1;
    delete[] h_C1;
    delete[] h_D1;
    
    delete[] h_A2;
    delete[] h_B2;
    delete[] h_C2;
    delete[] h_D2;
    
    delete[] h_A3;
    delete[] h_B3;
    delete[] h_C3;
    delete[] h_D3;

    delete[] h_A4;
    delete[] h_B4;
    delete[] h_C4;
    delete[] h_D4;
  
  return 0;
}
