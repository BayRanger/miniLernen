import torch
import MultiPyTorchCUDAKernel as multiple 

device = "cuda:0"

print(multiple.__file__)

if torch.cuda.is_available():
    # Allocate tensors on the GPU using torch.tensor method
    a = torch.ones((2,2), dtype=torch.float32, device=device)#.resize_(4)
    b = torch.ones((2,2), dtype=torch.float32, device=device)#.resize_(4)
   
    c = multiple.multi_gpu(a, b)
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c: {c}")