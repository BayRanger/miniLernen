#prefixes
# si prefixes
import time
import time
import numpy as np

N = 4096

if __name__=="__main__":
    dtype = np.float32
    A = np.random.randn(N, N).astype(dtype)
    B = np.random.randn(N, N).astype(dtype)

    flop = N * N * 2 *N
    print(f"{flop/1e9:.2f} GFLOPs")

    st = time.monotonic()

    C = A@B 
    
    et = time.monotonic()

    s = et - st
    print(f"{flop/s * 1e-12:.2f} TFLOP/s") 

    #https://stackoverflow.com/questions/10482974/why-is-stack-memory-size-so-limited 
