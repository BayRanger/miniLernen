#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>

namespace py = pybind11;

using T = float;
// declare templates for front (cpp) and back (cuda) sides of function:
void MultiGPUKernel(float *in_a, float *in_b, float *out_c, int N);

void MultiGPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c) {
  int N = in_a.numel();
  if (N != in_b.numel())
    throw std::invalid_argument("Size mismatch A.numel(): " + std::to_string(in_a.numel())
          + ", B.numel(): " + std::to_string(in_b.numel()));

  out_c.resize_({N});
  int n = in_a.sizes()[0];//row of ouput
  int m = in_b.sizes()[1];
  // call the kernel function...
  MultiGPUKernel(in_a.data_ptr<T>(), in_b.data_ptr<T>(),
          out_c.data_ptr<T>(), n);
  out_c.resize_({n,m});
}

void MultiGPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);

// declare the extension module with the AddGPU function:
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.doc() = "pybind11 example plugin";
  m.def("MultiGPU", &MultiGPU);
}