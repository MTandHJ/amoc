
import torch.nn as nn
from .base import AdversarialDefensiveModel





class Projector(AdversarialDefensiveModel):
    def __init__(self, dim_feature, dim_mlp, dim_head):
        super(Projector, self).__init__()

        self.head = nn.Sequential(
            nn.Linear(dim_feature, dim_mlp),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mlp, dim_head)
        )

    def forward(self, features):
        return self.head(features)


class FC(AdversarialDefensiveModel):
    def __init__(self, dim_feature, num_classes):
        super(FC, self).__init__()

        self.fc = nn.Linear(dim_feature, num_classes)
        
        # init the fc layer
        # self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()

    def forward(self, features):
        return self.fc(features)


class Wrapper(AdversarialDefensiveModel):

    def __init__(self, arch, fc):
        super(Wrapper, self).__init__()

        self.arch = arch
        self.fc = fc

    def forward(self, inputs):
        features = self.arch(inputs)
        outs = self.fc(features)
        return outs

        
