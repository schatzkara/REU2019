import torch

weights_path = 'C:/Users/Owner/Documents/UCF/Project/REU2019/weights/vgg16-397923af.pth'

state_dict = torch.load(weights_path)

for key, value in state_dict.items():
    print(key)
    first_per = key.index('.')
    second_per = key[first_per + 1:].index('.')
    id = key[:first_per + second_per + 1]
    print(id)
