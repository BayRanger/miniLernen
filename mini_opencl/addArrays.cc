#define CL_TARGET_OPENCL_VERSION 220  //opencl 2.2
#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define ARRAY_SIZE 100


// Kernel function to add two arrays element-wise
//set kernel file in a string, it is weird but it is how it works...
const char *kernelSource = 
"__kernel void addArrays(__global const float *a, __global const float *b, __global float *result) {\n"
"    int i = get_global_id(0);\n"
"    result[i] = a[i] + b[i];\n"
"}\n";

int main() {
    printf(">>> Initializing OpenCL...\n");
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufferA, bufferB, bufferResult;
    cl_int err;

    // Initialize OpenCL
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    // Create and compile the program
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Create kernel
    kernel = clCreateKernel(program, "addArrays", &err);

    // Create memory buffers
    float a[ARRAY_SIZE], b[ARRAY_SIZE], result[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, a, &err);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * ARRAY_SIZE, b, &err);
    bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * ARRAY_SIZE, NULL, &err);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);

    // Execute the kernel
    size_t globalWorkSize = ARRAY_SIZE;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

    // Read the result
    clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, sizeof(float) * ARRAY_SIZE, result, 0, NULL, NULL);

    // Print the result
    printf("Result:\n");
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%f ", result[i]);
    }
    printf("\n");

    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
