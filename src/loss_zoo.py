



import torch
import torch.nn as nn
import torch.nn.functional as F





def cross_entropy(outs, labels, reduction="mean"):
    return F.cross_entropy(outs, labels, reduction=reduction)

def kl_divergence(logits, targets, reduction="batchmean"):
    # KL divergence
    assert logits.size() == targets.size()
    inputs = F.log_softmax(logits, dim=-1)
    targets = F.softmax(targets, dim=-1)
    return F.kl_div(inputs, targets, reduction=reduction)

