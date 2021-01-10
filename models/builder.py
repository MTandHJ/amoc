
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base import AdversarialDefensiveModel


class MoCo(AdversarialDefensiveModel):


    def __init__(
        self, base_encoder, base_projector,
        dim_feature, dim_mlp, dim_head,
        momentum, T, K1, K2
    ):
        """Arguments:
        K1: the number of keys of queue clean
        K2: the number of keys of queue adv
        """
        super(MoCo, self).__init__()

        self.T = math.sqrt(T)
        self.K1 = K1
        self.K2 = K2
        self.m = momentum

        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()
        self.projector_q = base_projector(dim_feature, dim_mlp, dim_head)
        self.projector_k = base_projector(dim_feature, dim_mlp, dim_head)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad_(False)

        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad_(False)

        self.register_buffer('queue_clean', torch.randn(dim_head, K1))
        self.queue_clean = F.normalize(self.queue_clean, dim=0)
        self.register_buffer('queue_clean_ptr', torch.zeros(1, dtype=torch.long))

        self.register_buffer('queue_adv', torch.randn(dim_head, K2))
        self.queue_adv = F.normalize(self.queue_adv, dim=0)
        self.register_buffer('queue_adv_ptr', torch.zeros(1, dtype=torch.long))

    
    def momentum_update_key_encoder(self):
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data
        
        for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
            param_k.data = self.m * param_k.data + (1 - self.m) * param_q.data

    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.size(0)
        if self.adv_training:
            ptr = self.queue_adv_ptr[0]
            idx = torch.arange(ptr, ptr+batch_size) % self.K2
            self.queue_adv[:, idx] = keys.t()
            self.queue_adv_ptr[0] = idx[-1] + 1
        else:
            ptr = self.queue_clean_ptr[0]
            idx = torch.arange(ptr, ptr+batch_size) % self.K1
            self.queue_clean[:, idx] = keys.t()
            self.queue_clean_ptr[0] = idx[-1] + 1
            
    @property
    def qc(self):
        return self.queue_clean.clone().detach() / self.T

    @property
    def qa(self):
        return self.queue_adv.clone().detach() / self.T

    def forward_q(self, inputs_q):
        query_features = F.normalize(self.projector_q(self.encoder_q(inputs_q)), dim=1)
        return query_features / self.T

    def forward_k(self, inputs_k):
        with torch.no_grad():
            key_features = F.normalize(self.projector_k(self.encoder_k(inputs_k)), dim=1)
        if self.training:
            self._dequeue_and_enqueue(key_features)
        return key_features / self.T

    def forward(self, inputs):
        raise NotImplementedError("...")

            
class MoCoCCC(MoCo): pass

class MoCoCAC(MoCo):

    def forward_k(self, inputs_k):
        if self.attacking:
            key_features = F.normalize(self.projector_k(self.encoder_k(inputs_k)), dim=1)
            key_features = key_features / self.T
        else:
            key_features = super(MoCoCAC, self).forward_k(inputs_k)
        return key_features

    def forward(self, inputs):
        if self.attacking:
            inputs_q, inputs_k = inputs.split(inputs.size(0) // 2)
            self.adv_train(False)
            with torch.no_grad():
                outs_q = self.forward_q(inputs_q)
            self.adv_train(True)
            outs_k = self.forward_k(inputs_k)
            logits_pos = (outs_q * outs_k).sum(dim=1, keepdim=True)
            logits_neg = outs_q @ self.qc
            logits = torch.cat((logits_pos, logits_neg), dim=1)
            return logits
        else:
            raise NotImplementedError("Use forward_q, forward_k directly...")


class MoCoCAA(MoCoCAC):

    def forward(self, inputs):
        if self.attacking:
            inputs_q, inputs_k = inputs.split(inputs.size(0) // 2)
            self.adv_train(False)
            with torch.no_grad():
                outs_q = self.forward_q(inputs_q)
            self.adv_train(True)
            outs_k = self.forward_k(inputs_k)
            logits_pos = (outs_q * outs_k).sum(dim=1, keepdim=True)
            logits_neg = outs_q @ self.qa
            logits = torch.cat((logits_pos, logits_neg), dim=1)
            return logits
        else:
            raise NotImplementedError("Use forward_q, forward_k directly...")


class MoCoACC(MoCo):

    def forward(self, inputs):
        if self.attacking:
            inputs_q, inputs_k = inputs.split(inputs.size(0) // 2)
            self.adv_train(True)
            outs_q = self.forward_q(inputs_q)
            self.adv_train(False)
            outs_k = self.forward_k(inputs_k)
            logits_pos = (outs_q * outs_k).sum(dim=1, keepdim=True)
            logits_neg = outs_q @ self.qc
            logits = torch.cat((logits_pos, logits_neg), dim=1)
            return logits
        else:
            raise NotImplementedError("Use forward_q, forward_k directly...")

class MoCoACA(MoCo):

    def forward(self, inputs):
        if self.attacking:
            inputs_q, inputs_k = inputs.split(inputs.size(0) // 2)
            self.adv_train(True)
            outs_q = self.forward_q(inputs_q)
            self.adv_train(False)
            outs_k = self.forward_k(inputs_k)
            logits_pos = (outs_q * outs_k).sum(dim=1, keepdim=True)
            logits_neg = outs_q @ self.qa
            logits = torch.cat((logits_pos, logits_neg), dim=1)
            return logits
        else:
            raise NotImplementedError("Use forward_q, forward_k directly...")

class MoCoACA_PLUS(MoCoACA): pass

class MoCoAAC(MoCoCAC):

    def forward(self, inputs):
        if self.attacking:
            inputs_q, inputs_k = inputs.split(inputs.size(0) // 2)
            self.adv_train(True)
            outs_q, outs_k = self.forward_q(inputs_q), self.forward_k(inputs_k)
            logits_pos = (outs_q * outs_k).sum(dim=1, keepdim=True)
            logits_neg = outs_q @ self.qc
            logits = torch.cat((logits_pos, logits_neg), dim=1)
            return logits
        else:
            raise NotImplementedError("Use forward_q, forward_k directly...")

class MoCoAAA(MoCoAAC):

    def forward(self, inputs):
        if self.attacking:
            inputs_q, inputs_k = inputs.split(inputs.size(0) // 2)
            self.adv_train(True)
            outs_q, outs_k = self.forward_q(inputs_q), self.forward_k(inputs_k)
            logits_pos = (outs_q * outs_k).sum(dim=1, keepdim=True)
            logits_neg = outs_q @ self.qa
            logits = torch.cat((logits_pos, logits_neg), dim=1)
            return logits
        else:
            raise NotImplementedError("Use forward_q, forward_k directly...")
        
