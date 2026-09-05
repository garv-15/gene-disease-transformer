import torch
import torch.nn as nn

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import torch


a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])


if torch.backends.mps.is_available():
    a = a.to("mps")
    b= b.to("mps")

print(a + b)
print(a * b)
print(a.sum())
print(a.mean())

print(f"current device: {device}")
