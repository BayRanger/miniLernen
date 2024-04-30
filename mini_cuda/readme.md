when we talk about cuda programming, it is always suffering to read some code similar to normal c/cpp code, but with some special definitions, like kernel, block, etc...

Let's figure it out.

## Compile

```
nvcc matrix_multi.cu -o matrix_multi
```

this code could helps to check if your cuda compiler works for further steps...

## Requirements

install cuda, and check it via

`nvcc -V`

## Run

`./matrix_multi`

## Further learning

This lecture has a good note!

https://users.wfu.edu/choss/CUDA/docs/
