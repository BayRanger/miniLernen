#include <cuda_runtime.h>

__global__ void tranposeNative(float* input, float *output, int m, int n)
{
    int col1D_input = threadIdx.x + blockDim.x * blockIdx.x;
    int row1D_input = threadIdx.y + blockDim.y * blockIdx.y;

    if (row1D_input < m && col1D_input <n)
    {
        int index_input = col1D_input + row1D_input * n;
        int index_output = row1D_input + col1D_input * n;

        output[index_output] = input[index_input];
    }
}


__global__  void tranposeOptimized(float *input, float *ouptut, int m, int n)
{
    int col1D_input = threadIdx.x + blockDim.x * blockIdx.x;
    int row1D_input = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float sdata[32][33];

    if (row1D_input < m && col1D_input <n)
    {
        int index_input = col1D_input + row1D_input * n;
        //int index_output = row1D_input + col1D_input * n;
        sdata[threadIdx.y][threadIdx.x] = input[index_input];

        __syncthreads();
       int dst_col = threadIdx.x + blockIdx.y + blockDim.y;
       int dst_row = threadIdx.y + blockIdx.x + blockDim.x;

        output[dst_col+ dst_row *n] = sdata[threadIdx.x][threadIdx.y];
    }
}



int main()
{

    int m =8192;
    int n = 4096;
        // Host matrices
    float *h_input = new float[m * n];
    float *h_output = new float[m * n];

    // Initialize input matrix with some values
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            h_input[i * n + j] = static_cast<float>(i * n + j);
        }
    }

    float *d_input, *d_output;
    // initialize matrices a and b with appropriate values
    int size = m * n * sizeof(int);
    //allocate memory in GPU, it modify the addr
    //https://forums.developer.nvidia.com/t/void-d-why-two/12609
    cudaMalloc((void **) &d_input, size);
    cudaMalloc((void **) &d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((n+ blockDim.x-1)/blockDim.x, (m+ blockDim.y-1)/blockDim.y);

    tranposeNative<<<gridDim,blockDim >>>(d_input, d_output, m, n);
   
    // Copy device memory to host
    cudaMemcpy(h_output, d_output, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;
    delete[] h_output;
}


