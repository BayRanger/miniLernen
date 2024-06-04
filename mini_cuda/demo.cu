/**
 * basic code to cun cuda multiplication
 * 
 * TODO: benchmark the time duration 
 * 
 * this code is enough  to learn key cuda operation and data structure.
 * 
 * https://users.wfu.edu/choss/CUDA/docs/Lecture%205.pdf
*/

#define N 100

#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}


__global__ void vecAdd(int* a, int* b, int* c, int n)
{
    int id = blockIdx.x* blockDim.x + threadIdx.x;

    if (id < n)
        c[id] = a[id] + b[id];
}


__global__ void matrixMultCuda (int *a, int *b, int *c, int width) 
{
     int k, sum = 0;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if(col < width && row < width) 
    {
        for (k = 0; k < width; k++)
        sum += a[row * width + k] * b[k * width + col];
    
    c[row * width + col] = sum;
        
    }
}


void matrixMult (int a[N][N], int b[N][N], int c[N][N], int width)
{
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) 
        {
            int sum = 0;
            for (int k = 0; k < width; k++)
            {
                int m = a[i][k];
                int n = b[k][j];
                sum += m * n;
            }
            c[i][j] = sum;
        }
    }
}
void test_GEMM() {
    int a[N][N], b[N][N], c[N][N];
    int *dev_a, *dev_b, *dev_c;
    // initialize matrices a and b with appropriate values
    int size = N * N * sizeof(int);
    //allocate memory in GPU, it modify the addr
    //https://forums.developer.nvidia.com/t/void-d-why-two/12609
    cudaMalloc((void **) &dev_a, size);
    cudaMalloc((void **) &dev_b, size);
    cudaMalloc((void **) &dev_c, size);
    //copy data from cpu to gpu
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);
    //definme the grid dimension : how many blocks a grid has
    //define the block dimension : how many threads f block has
    dim3 dimGrid(1, 1);
    dim3 dimBlock(N, N);

    //Execute kernel!
    matrixMultCuda<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
    //perform calcualtion and copy back data to host
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    //free gpu memory acclocation
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    //cpu version matrix multi
    matrixMult(a, b, c, N);
}

void  test_vecAdd()
{
    int num  = 1<<10;
    //the size means SIZE in BYTE
    int size = num* sizeof(int);
    int a[num], b[num], c[num], c_gt[num];
    for (int i=0; i<num; i++)
    {
        a[i] = b[i] = i;
        c_gt[i] = a[i] + b[i];
        
    }
    int *dev_a, *dev_b, *dev_c;
    //allocate memory in GPU
    cudaMalloc((void **) &dev_a, size);
    cudaMalloc((void **) &dev_b, size);
    cudaMalloc((void **) &dev_c, size);
    //copy data from cpu to gpu
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b,size , cudaMemcpyHostToDevice);
    //definme the grid dimension : how many blocks a grid has
    //define the block dimension : how many threads f block has
    int blockSize, gridSize;
      
    // Number of threads in each thread block
    blockSize = 256;
      
    // Number of thread blocks in grid
    gridSize = ceil((float)num/blockSize);
    // Execute the kernel
    vecAdd<<< gridSize, blockSize >>>(dev_a, dev_b, dev_c, num);
    //perform calcualtion and copy back data to host
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    //free gpu memory acclocation
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    
    for (int i=0; i <num; i++)
    {
        std::cout<<"check "<<c[i]<<" vs "<<c_gt[i]<<std::endl;
        if(c[i] - c_gt[i]!=0)
        {
            std::cout<<"VectorAdd wrong\n";
            break;
        }
        else if(i == num -1)
        {
            std::cout<<"vector add success!\n";
        }
    }
}


void test_big_vecadd()
{
    int numElements = 1<<20;
    size_t size = numElements * sizeof(float);


    //Allocate host memory
    float* h_A = (float *)malloc(size);
    float* h_B = (float *)malloc(size);
    float* h_C = (float *)malloc(size);

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }                     
       // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size); 

    // Copy the host input vectors to the device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);


    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256 *2 ;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;


    int numIterations = 100;
    float totalMilliseconds = 0.0;

    for (int i = 0; i < numIterations; ++i) {
        // Start the timer
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

        // Stop the timer
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalMilliseconds += milliseconds;

        // Destroy the events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
   
    // Calculate the average time and performance in TOPS
    float averageMilliseconds = totalMilliseconds / numIterations;
    float averageSeconds = averageMilliseconds / 1000.0;
    float averageTops = (numElements / averageSeconds) / 1e12;
    printf("Average Performance: %.2f TOPS\n", averageTops);
 
    
    // Copy the device result vector to the host result vector
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


    // Free device global memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return;

}


int main()
{
	test_big_vecadd();
}
