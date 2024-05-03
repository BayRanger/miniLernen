# Introduction

OpenCL came as a **standard** for **heterogeneous programming** that enables a code to run in different platforms, such as multicore CPUs, GPUs (AMD, Intel, ARM), FPGAs, Apple M1, tensor cores, and ARM processors with minor or no modifications.

Furthermore, differently from OpenACC, the programmer has **full control** of the hardware and is **entirely** responsible for the parallelization process.

# Run

### install

sudo apt install opencl-headers ocl-icd-opencl-dev -y

### comile & run

```
gcc -o addArrays addArrays.cc -lOpenCL

```

./addArrays

# Reference

https://github.com/gcielniak/OpenCL-Tutorials/tree/master

https://ulhpc-tutorials.readthedocs.io/en/latest/gpu/opencl/#:~:text=Compiling%20an%20OpenCL%20code%3A,flag%20to%20the%20compilation%20command.

https://github.com/KhronosGroup/OpenCL-Guide/blob/main/chapters/getting_started_linux.md

https://github.com/olcf/vector_addition_tutorials

https://www.olcf.ornl.gov/tutorials/opencl-vector-addition/
