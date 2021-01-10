
import torch
import numpy as np
import os
from PIL import Image
from .config import ROOT


# https://github.com/VITA-Group/Adversarial-Contrastive-Learning/blob/937019219497b449f4cb61cc6118fdf32cc3de12/data/cifar10_c.py#L9
class CIFAR10C(torch.utils.data.Dataset):
    filename = "CIFAR-10-C"
    def __init__(self, root=ROOT, transform=None, severity=5, corruption_type=''):
        root = os.path.join(root, self.filename)
        dataPath = os.path.join(root, '{}.npy'.format(corruption_type))
        labelPath = os.path.join(root, 'labels.npy')

        self.data = np.load(dataPath)
        self.label = np.load(labelPath).astype(np.long)
        self.transform = transform

    def __getitem__(self, idx):

        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = self.label[idx]

        return img, label

    def __len__(self):
        return self.data.shape[0]
        

class CIFAR100C(CIFAR10C):
    filename = "CIFAR-100-C"


class SingleSet(torch.utils.data.Dataset):

    def __init__(self, dataset, transform):
        self.data = dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, label = self.data[index]
        return self.transform(img), label

class DoubleSet(SingleSet):

    def __getitem__(self, index):
        img, label = self.data[index]
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2, label
