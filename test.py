import torch

# print(5 * torch.ones(2, 2))
#
# app = torch.ones(1, 2, 7, 7)

# bsz, channels, height, width = app.size()
# buffer = torch.zeros(bsz, channels, 8, height, width)
# print(app.size())
# print(buffer.size())
# for frame in range(8):
#     buffer[:, :, frame, :, :] = app
#
# print(buffer)
# print(buffer.size())

print(torch.tensor([5]) - torch.tensor([2]))

