


import torch
import eagerpy as ep
import foolbox as fb
from foolbox.criteria import Criterion, Misclassification, TargetedMisclassification


class TransferClassification(Misclassification):

    def __init__(self, model, labels):
        super(TransferClassification, self).__init__(labels)
        self.target_model = model
    
    def __call__(self, perturbed, outputs):
        outputs = self.target_model(perturbed)
        return super().__call__(perturbed, outputs)


class DoubleFalse(Misclassification):

    def __init__(self, labels):
        super(DoubleFalse, self).__init__(labels)
        n = labels.size(0)
        all_false = torch.zeros(n * 2, dtype=torch.bool).to(labels.device)
        self.all_false = ep.astensor(all_false)

    def __call__(self, perturbed, outputs):
        return self.all_false
        

class LogitsAllFalse(Misclassification):

    def __init__(self, logits):
        super(LogitsAllFalse, self).__init__(logits)
        n = logits.size(0)
        all_false = torch.zeros(n, dtype=torch.bool).to(logits.device)
        self.all_false = ep.astensor(all_false)

    def __call__(self, perturbed, outputs):
        return self.all_false