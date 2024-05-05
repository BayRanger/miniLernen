# CUDA Kernel for PyTorch

This codebase demonstrates how to:

- implement a simple CUDA kernel function
- interface it to PyTorch using pybind11 / ATen

To Build and install

```
- python3 setup.py install --force
```

To test

```
python test.py
```


After build, you will:

* find a **package**  named **MultiPyTorchCUDAKernel**[suffix] in you **site-packages** folder
* use **dummMultiply** as a module name to import and use
* the kernel interface named **MultiGPU**
* a more friendly interfece named **multi_gpu**



## TODO

it is a very naive implementation of multiply using cuda kernel, but at least it is a cuda kernel
To improve it, you could:

* make it robust to different matrix size
* more reasonable kernel size setting
* compatible with different dtype (only float is allowd now)

## Reference

https://github.com/miramar-labs/jetson-nano-dev
