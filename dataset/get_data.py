import torch
import torchvision
import os
from torchvision import transforms, datasets



def get_dataloader(data_dir, input_dir, batch_size, train=True):
    imgs = datasets.ImageFolder(os.path.join(data_dir, input_dir),
                                            transform=transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.RandomCrop(227),
                                                    transforms.ToTensor(),
                                            ]))

    if train:
        data_loader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=True)
    else:
        data_loader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=False)
    return data_loader