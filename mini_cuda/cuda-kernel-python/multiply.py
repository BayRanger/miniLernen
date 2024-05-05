import torch

from DummMultiply import MultiGPU

def multi_gpu(a, b):
    assert isinstance(a, torch.cuda.FloatTensor) 
    assert isinstance(b, torch.cuda.FloatTensor)
    assert a.numel() == b.numel()

    c = a.new()
    MultiGPU(a, b, c)
    return c