import numpy as np
import torch

a = [1, 2, 3]
b = [0, 1, 2]

a = np.array(a)
b = np.array(b)

print(b - a)
print(np.sum(np.abs(b - a)))

a = torch.randn(2, 2, 2, 2)
a = torch.unsqueeze(a, dim=2)
print(a.size())
