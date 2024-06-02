/**
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
#include <iostream>

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
    //the size means SIZE in BYTE
    int size = N* sizeof(int);
    int a[N], b[N], c[N], c_gt[N];
    for (int i=0; i<N; i++)
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
    gridSize = ceil((float)N/blockSize);
    // Execute the kernel
    vecAdd<<< gridSize, blockSize >>>(dev_a, dev_b, dev_c, N);
    //perform calcualtion and copy back data to host
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    //free gpu memory acclocation
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    
    for (int i=0; i <N; i++)
    {
        std::cout<<"check "<<c[i]<<" vs "<<c_gt[i]<<std::endl;
        if(c[i] - c_gt[i]!=0)
        {
            std::cout<<"VectorAdd wrong\n";
            break;
        }
        else if(i == N -1)
        {
            std::cout<<"vector add success!\n";
        }
    }
}

int main()
{

    test_vecAdd();
}
