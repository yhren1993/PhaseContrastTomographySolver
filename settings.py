import torch

bit = 32

np_float_datatype      = "float32" if bit == 32 else "float64"
np_complex_datatype    = "complex64" if bit == 32 else "complex128"
torch_float_datatype   = torch.FloatTensor if bit == 32 else torch.DoubleTensor
