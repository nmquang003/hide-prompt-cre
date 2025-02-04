from methods.ct_loss import constractive_loss, sim
import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)

b = torch.tensor([[[1, 2, 3], [1, 2, 1]], [[4, 5, 0], [4, 5, 3]], [[7, 8, 9], [1, 2, 3]]], dtype=torch.float32)

c = torch.tensor([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]], dtype=torch.float32)

print(sim(a, b))

print(constractive_loss(a, b, c))
