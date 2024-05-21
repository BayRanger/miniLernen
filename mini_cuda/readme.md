when we talk about cuda programming, it is always suffering to read some code similar to normal c/cpp code, but with some special definitions, like kernel, block, etc...

So, let's figure it out step by step

* step 1: write a simple cuda kernel and run it
* step 2: use the cuda kernel in your python code
* step 3: integrate it into tensorrt inference engine

there are two examples provided in the code, namely vector add and matrix multiply

Let's figure it out.

## Compile

```
nvcc demo.cu -o demo 
```

this code could helps to check if your cuda compiler works for further steps...



## Requirements

install cuda, and check it via

`nvcc -V`

## Run

`./demo`

## Further learning

good blog

https://siboehm.com/articles/22/CUDA-MMM

This lecture has a good note!

https://users.wfu.edu/choss/CUDA/docs/

for vector add

https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/
