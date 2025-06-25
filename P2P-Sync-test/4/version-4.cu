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
    double *h_A1 = new double[N];
    double *h_B1 = new double[N];
    double *h_C1 = new double[N];
    double *h_D1 = new double[N];
    
    double *h_A2 = new double[N];
    double *h_B2 = new double[N];
    double *h_C2 = new double[N];
    double *h_D2 = new double[N];
    
    double *h_A3 = new double[N];
    double *h_B3 = new double[N];
    double *h_C3 = new double[N];
    double *h_D3 = new double[N];
        
    double *h_A4 = new double[N];
    double *h_B4 = new double[N];
    double *h_C4 = new double[N];
    double *h_D4 = new double[N];
    
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
    

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    for (int i = 0; i < deviceCount; ++i) {
      cudaSetDevice(i); // Set GPU i as active
      for (int j = 0; j < deviceCount; ++j) {
        if (i == j) continue; // Skip self-access
        int canAccess = 0;
        cudaDeviceCanAccessPeer(&canAccess, i, j);
        if (canAccess) {
	  cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
	  if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
	    printf("P2P enable failed: GPU %d -> GPU %d: %s\n", i, j, cudaGetErrorString(err));
	  }
        }
      }
    }
    
    // Allocate device memory
    double *d_A1 = nullptr, *d_B1 = nullptr, *d_C1 = nullptr, *d_D1 = nullptr;
    double *d_A2 = nullptr, *d_B2 = nullptr, *d_C2 = nullptr, *d_D2 = nullptr;
    double *d_A3 = nullptr, *d_B3 = nullptr, *d_C3 = nullptr, *d_D3 = nullptr;
    double *d_A4 = nullptr, *d_B4 = nullptr, *d_C4 = nullptr, *d_D4 = nullptr;

    cudaSetDevice(0);
    checkCudaError(cudaMalloc(&d_A1, bytes), "cudaMalloc d_A1");
    checkCudaError(cudaMalloc(&d_B1, bytes), "cudaMalloc d_B1");
    checkCudaError(cudaMalloc(&d_C1, bytes), "cudaMalloc d_C1");
    checkCudaError(cudaMalloc(&d_D1, bytes), "cudaMalloc d_D1");
    cudaMemcpy(d_A1, h_A1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B1, h_B1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1, h_C1, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D1, h_D1, bytes, cudaMemcpyHostToDevice);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaDeviceEnablePeerAccess(2, 0);
    cudaDeviceEnablePeerAccess(3, 0);

    cudaSetDevice(1);
    checkCudaError(cudaMalloc(&d_A2, bytes), "cudaMalloc d_A2");
    checkCudaError(cudaMalloc(&d_B2, bytes), "cudaMalloc d_B2");
    checkCudaError(cudaMalloc(&d_C2, bytes), "cudaMalloc d_C2");
    checkCudaError(cudaMalloc(&d_D2, bytes), "cudaMalloc d_D2");
    cudaMemcpy(d_A2, h_A2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, h_B2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2, h_C2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D2, h_D2, bytes, cudaMemcpyHostToDevice);
    cudaDeviceEnablePeerAccess(0, 0);
    cudaDeviceEnablePeerAccess(2, 0);
    cudaDeviceEnablePeerAccess(3, 0);
    
    cudaSetDevice(2);
    checkCudaError(cudaMalloc(&d_A3, bytes), "cudaMalloc d_A3");
    checkCudaError(cudaMalloc(&d_B3, bytes), "cudaMalloc d_B3");
    checkCudaError(cudaMalloc(&d_C3, bytes), "cudaMalloc d_C3");
    checkCudaError(cudaMalloc(&d_D3, bytes), "cudaMalloc d_D3");
    cudaMemcpy(d_A3, h_A3, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B3, h_B3, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C3, h_C3, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D3, h_D3, bytes, cudaMemcpyHostToDevice);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaDeviceEnablePeerAccess(3, 0);
    cudaDeviceEnablePeerAccess(0, 0);
      
    cudaSetDevice(3);
    checkCudaError(cudaMalloc(&d_A4, bytes), "cudaMalloc d_A4");
    checkCudaError(cudaMalloc(&d_B4, bytes), "cudaMalloc d_B4");
    checkCudaError(cudaMalloc(&d_C4, bytes), "cudaMalloc d_C4");
    checkCudaError(cudaMalloc(&d_D4, bytes), "cudaMalloc d_D4");
    cudaMemcpy(d_A4, h_A4, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B4, h_B4, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C4, h_C4, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D4, h_D4, bytes, cudaMemcpyHostToDevice);
    cudaDeviceEnablePeerAccess(2, 0);
    cudaDeviceEnablePeerAccess(0, 0);
    cudaDeviceEnablePeerAccess(1, 0);


    int numberofgpus = 4;      
    cudaStream_t halo_stream[numberofgpus];
    
    // create a streams 
    for (int i = 0; i < numberofgpus; ++i) 
      {
	cudaSetDevice(i);
	cudaStreamCreate(&halo_stream[i]);
      }
    
    omp_set_num_threads(3);
    
    double start = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp sections
      {
#pragma omp section
	{
	  cudaSetDevice(1);
	  checkCudaError(cudaMemcpyPeer(d_B1, 0, d_B2, 1, bytes), "cudaMemcpy d_A");
	  //	  cudaStreamSynchronize(halo_stream[1]);
	  cudaDeviceSynchronize();
	}
#pragma omp section
	{
	  cudaSetDevice(2);
	  checkCudaError(cudaMemcpyPeer(d_C1, 0, d_C3, 2, bytes), "cudaMemcpy d_A");
	  //	  cudaStreamSynchronize(halo_stream[2]);
	  cudaDeviceSynchronize();
	}
#pragma omp section
	{
	  cudaSetDevice(3);
	  checkCudaError(cudaMemcpyPeer(d_D1, 0, d_D4, 3, bytes), "cudaMemcpy d_A");
	  //	  cudaStreamSynchronize(halo_stream[3]);
	  cudaDeviceSynchronize();
	}	
      }
    }        
    double end = omp_get_wtime();
    double milliseconds = (end - start) * 1000.0;  
    
    // Convert to seconds and compute bandwidth (2 copies)
    double seconds = milliseconds / 1000.0;
    double total_bytes = 3*bytes; // two transfers: h_A → d_A and h_B → d_B
    double bandwidthGBps = total_bytes / (seconds * 1e9);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Elapsed time: " << seconds << " seconds\n";
    std::cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    std::cout << "PCIe Bandwidth: " << bandwidthGBps << " GB/s\n";


    cudaSetDevice(1);
    cudaMemcpy(h_B2, d_B2, bytes, cudaMemcpyDeviceToHost);
    cudaSetDevice(2);
    cudaMemcpy(h_C3, d_C3, bytes, cudaMemcpyDeviceToHost);
    cudaSetDevice(3);
    cudaMemcpy(h_D4, d_D4, bytes, cudaMemcpyDeviceToHost);

    /*    
    for (int i = 0; i < N; ++i) {
      std::cout << "h_B2[" << i << "] = " << h_B2[i] << std::endl;
    }
    
    for (int i = 0; i < N; ++i) {
      std::cout << "h_C3[" << i << "] = " << h_C3[i] << std::endl;
    }

    for (int i = 0; i < N; ++i) {
      std::cout << "h_D4[" << i << "] = " << h_D4[i] << std::endl;
    }
    */
    
    // Clean up
    cudaSetDevice(0);
    checkCudaError(cudaFree(d_A1), "cudaFree d_A1");
    checkCudaError(cudaFree(d_B1), "cudaFree d_B2");
    checkCudaError(cudaFree(d_C1), "cudaFree d_A3");
    checkCudaError(cudaFree(d_D1), "cudaFree d_B");
    
    cudaSetDevice(1);
    checkCudaError(cudaFree(d_A2), "cudaFree d_A2");
    checkCudaError(cudaFree(d_B2), "cudaFree d_B2");
    checkCudaError(cudaFree(d_C2), "cudaFree d_C2");
    checkCudaError(cudaFree(d_D2), "cudaFree d_D2");
    
    cudaSetDevice(2);
    checkCudaError(cudaFree(d_A3), "cudaFree d_A3");
    checkCudaError(cudaFree(d_B3), "cudaFree d_B3");
    checkCudaError(cudaFree(d_C3), "cudaFree d_C3");
    checkCudaError(cudaFree(d_D3), "cudaFree d_D3");
    
    cudaSetDevice(3);
    checkCudaError(cudaFree(d_A4), "cudaFree d_A4");
    checkCudaError(cudaFree(d_B4), "cudaFree d_B4");
    checkCudaError(cudaFree(d_C4), "cudaFree d_C4");
    checkCudaError(cudaFree(d_D4), "cudaFree d_D4");

       
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
    
    return EXIT_SUCCESS;
}
