# Descirption

this code shows how to use cublas to calculate maximum and minimum of the list


# Introduction

CUBLAS is an abbreviation for CUDA Basic Linear Algebra Subprograms.


## Run

```
Two options to compile file and generate executable file ismax


 g++ ismax.c -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcuda -lcublas -lcudart -o ismax

```

```
nvcc ismax.c --verbose -lcublas -o ismax 
```


# Further learning

https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf

http://courses.cms.caltech.edu/cs179/2023_lectures/cs179_2023_lec10.pdf
