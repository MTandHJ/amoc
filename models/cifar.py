


import torch
import torch.nn as nn
from .base import AdversarialDefensiveModel, TriggerBN1d, TriggerBN2d, Sequential




class CIFAR(AdversarialDefensiveModel):

    def __init__(self, dim_feature=256):
        super(CIFAR, self).__init__()

        self.conv = Sequential(  # 3 x 32 x 32
            nn.Conv2d(3, 64, 3),  # 64 x 30 x 30
            TriggerBN2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),  # 64 x 28 x 28
            TriggerBN2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 14 x 14
            nn.Conv2d(64, 128, 3),  # 128 x 12 x 12
            TriggerBN2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),  # 128 x 10 x 10
            TriggerBN2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128 x 5 x 5
        )

        self.dense = Sequential(
            nn.Linear(128 * 5 * 5, 256),
            TriggerBN1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim_feature),
            TriggerBN1d(dim_feature)
        )
        self.activation = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.conv(x).flatten(start_dim=1)
        features = self.activation(self.dense(x))
        return features

