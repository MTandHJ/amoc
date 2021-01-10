




import torch.nn as nn
import abc

class ADType(abc.ABC): ...

class AdversarialDefensiveModel(ADType, nn.Module):
    """
    Define some basic properties.
    """
    def __init__(self):
        super(AdversarialDefensiveModel, self).__init__()
        self.adv_training = False
        self.attacking = False

        
    def adv_train(self, mode=True):
        # enter contrastive mode
        self.adv_training = mode
        for module in self.children():
            if isinstance(module, ADType):
                module.adv_train(mode)

    def attack(self, mode=True):
        # enter contrastive mode
        self.attacking = mode
        for module in self.children():
            if isinstance(module, ADType):
                module.attack(mode)

class Sequential(nn.Sequential, AdversarialDefensiveModel): pass

class TriggerBN1d(AdversarialDefensiveModel):
    
    def __init__(self, num_features):
        super(TriggerBN1d, self).__init__()
        self.bn_clean = nn.BatchNorm1d(num_features)
        self.bn_adv = nn.BatchNorm1d(num_features)

    def forward(self, x):
        if self.adv_training:
            return self.bn_adv(x)
        else:
            return self.bn_clean(x)

class TriggerBN2d(AdversarialDefensiveModel):
    
    def __init__(self, num_features):
        super(TriggerBN2d, self).__init__()
        self.bn_clean = nn.BatchNorm2d(num_features)
        self.bn_adv = nn.BatchNorm2d(num_features)

    def forward(self, x):
        if self.adv_training:
            return self.bn_adv(x)
        else:
            return self.bn_clean(x)


if __name__ == "__main__":
    
    model = AdversarialDefensiveModel()
    model.child1 = AdversarialDefensiveModel()
    model.child2 = AdversarialDefensiveModel()

    print(model.adv_training)
    model.adv_train(True)
    for m in model.modules():
        print(m.adv_training)
    

