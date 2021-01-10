



import torch
import eagerpy as ep
from foolbox.attacks import LinfProjectedGradientDescentAttack


class LinfPGDKLDiv(LinfProjectedGradientDescentAttack):

    def get_loss_fn(self, model, logits_p):
        def loss_fn(inputs):
            logits_q = model(inputs)
            return ep.kl_div_with_logits(logits_p, logits_q).sum()
        return loss_fn









