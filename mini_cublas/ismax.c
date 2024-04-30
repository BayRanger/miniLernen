// nvcc 001 isamax .c -lcublas
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#define n 6 // length of x
int main ( void ){
    cudaError_t cudaStat ; // cudaMalloc status
    cublasStatus_t stat ; // CUBLAS functions status
    cublasHandle_t handle ; // CUBLAS context
    int j; // index of elements
    float * x; // n- vector on the host
    x=( float *) malloc (n* sizeof (*x)); // host memory alloc
    for(j=0;j<n;j++)
    x[j]=( float )j; // x={0 ,1 ,2 ,3 ,4 ,5}
    printf ("x: ");
    for(j=0;j<n;j++)
    printf (" %4.0f,",x[j]); // print x
    printf ("\n");
    // on the device
    float * d_x; // d_x - x on the device
    cudaStat = cudaMalloc (( void **)& d_x ,n* sizeof (*x)); // device
    // memory alloc for x
    stat = cublasCreate (& handle ); // initialize CUBLAS context
    stat = cublasSetVector (n, sizeof (*x) ,x ,1 ,d_x ,1); // cp x ->d_x
    int result ; // index of the maximal / minimal element
    // find the smallest index of the element of d_x with maximum
    // absolute value
    stat=cublasIsamax(handle,n,d_x,1,&result);
    printf ("max |x[i ]|:%4.0f\n",fabs (x[result -1])); // print
    // max {|x [0]| ,... ,| x[n -1]|}
    // find the smallest index of the element of d_x with minimum
    // absolute value
    stat=cublasIsamin(handle,n,d_x,1,&result);
    printf ("min |x[i ]|:%4.0f\n",fabs (x[result -1])); // print
    // min {|x [0]| ,... ,| x[n -1]|}
    cudaFree (d_x ); // free device memory
    cublasDestroy ( handle ); // destroy CUBLAS context
    free (x); // free host memory
    return EXIT_SUCCESS ;
}
// x: 0 , 1 , 2 , 3 , 4 , 5 ,
// max |x[i]|: 5
// min |x[i]|: 0