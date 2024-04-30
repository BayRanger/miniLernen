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
int main() {

    int a[N][N], b[N][N], c[N][N];
    int *dev_a, *dev_b, *dev_c;
    // initialize matrices a and b with appropriate values
    int size = N * N * sizeof(int);
    //allocate memory in GPU
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

    //this kernel!
    matrixMultCuda<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c, N);
    //perform calcualtion and copy back data to host
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);
    //free gpu memory acclocation
    cudaFree(dev_a); cudaFree(dev_b); cudaFree(dev_c);
    //cpu version matrix multi
    matrixMult(a, b, c, N);
}
