/*
 * basic code to cun cuda multiplication
 * 
 * TODO: benchmark the time duration 
 * 
 * this code is enough  to learn key cuda operation and data structure.
 * 
 * https://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
*/


#define N 16
#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>



__global__ void matrixMultCuda (float *a, float *b, float *c, int width) 
{
    int k = 0;
    float sum = 0;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if(col < width && row < width) 
    {
      for (k = 0; k < width; k++)
      sum += a[row * width + k] * b[k * width + col];
    c[row * width + col] = sum;
        
    }
}


// define the kernel calling code:

void MultiGPUKernel(float* dev_a, float* dev_b, float* dev_c, int n) {
  dim3 dimGrid(1, 1);
  dim3 dimBlock(N, N);
  matrixMultCuda<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, n);
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error("CUDA kernel failed : " + std::to_string(err));
}
