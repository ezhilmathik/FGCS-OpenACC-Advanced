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
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
	h_C[i] = 3.0;
	h_D[i] = 4.0;	
    }

    // Allocate device memory
    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr, *d_D = nullptr;
    cudaSetDevice(0);
    checkCudaError(cudaMalloc(&d_A, bytes), "cudaMalloc d_A");
    checkCudaError(cudaMalloc(&d_D, bytes), "cudaMalloc d_D");
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice);
    cudaDeviceEnablePeerAccess(1, 0);

    cudaSetDevice(1);
    checkCudaError(cudaMalloc(&d_B, bytes), "cudaMalloc d_B");
    checkCudaError(cudaMalloc(&d_C, bytes), "cudaMalloc d_C");
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);
    cudaDeviceEnablePeerAccess(0, 0);
    
    int numberofgpus = 2;      
    //    cudaStream_t compute_stream[numberofgpus];
    cudaStream_t halo_stream[numberofgpus];
    
    // create a streams 
    for (int i = 0; i < numberofgpus; ++i) 
      {
	cudaSetDevice(i);
	cudaStreamCreate(&halo_stream[i]);
      }

    omp_set_num_threads(2);
    
    /*    
    // Start timing
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");
    checkCudaError(cudaEventRecord(start), "cudaEventRecord start");
    */
    double start = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp sections
      {
#pragma omp section
	{
	  // Copy data to both GPUs
	  cudaSetDevice(0);
	  checkCudaError(cudaMemcpyPeer(d_C, 1, d_A, 0, bytes), "cudaMemcpy d_A");
	  //	  cudaStreamSynchronize(halo_stream[0]);
	  cudaDeviceSynchronize();
	}
#pragma omp section
	{
	  cudaSetDevice(1);
	  checkCudaError(cudaMemcpyPeer(d_D, 0, d_B, 1, bytes), "cudaMemcpy d_A");
	  //	  cudaStreamSynchronize(halo_stream[1]);
	  cudaDeviceSynchronize();
	}
      }
    }
    

    double end = omp_get_wtime();
    double milliseconds = (end - start) * 1000.0;  
    
    // Convert to seconds and compute bandwidth (2 copies)
    double seconds = milliseconds / 1000.0;
    double total_bytes = 2*bytes; // two transfers: h_A → d_A and h_B → d_B
    double bandwidthGBps = total_bytes / (seconds * 1e9);
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Elapsed time: " << seconds << " seconds\n";
    std::cout << "Total transferred: " << total_bytes / (1024.0 * 1024.0 * 1024.0) << " GB\n";
    std::cout << "PCIe Bandwidth: " << bandwidthGBps << " GB/s\n";


    cudaSetDevice(0);
    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);


    /*    
    for (int i = 0; i < N; ++i) {
      std::cout << "h_C[" << i << "] = " << h_C[i] << std::endl;
    }
    
    // Print the vector
    for (int i = 0; i < N; ++i) {
      std::cout << "h_D[" << i << "] = " << h_D[i] << std::endl;
    }
    */
    
    // Clean up
    cudaSetDevice(0);
    checkCudaError(cudaFree(d_A), "cudaFree d_A");
    checkCudaError(cudaFree(d_D), "cudaFree d_D");
    
    cudaSetDevice(1);
    checkCudaError(cudaFree(d_B), "cudaFree d_B");
    checkCudaError(cudaFree(d_C), "cudaFree d_C");

    
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_D;

    //    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    //    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");

    return EXIT_SUCCESS;
}
