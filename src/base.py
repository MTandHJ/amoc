


import torch
import torch.nn as nn
import foolbox as fb
import eagerpy as ep
from .utils import AverageMeter, ProgressMeter
from .criteria import DoubleFalse, LogitsAllFalse
from .loss_zoo import cross_entropy, kl_divergence
from models.builder import MoCo



def enter_attack_exit(func):
    def wrapper(attacker, *args, **kwargs):
        attacker.model.attack(True)
        results = func(attacker, *args, **kwargs)
        attacker.model.attack(False)
        return results
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class Coach:
    """
    Coach is used to train models.
    model: ...
    device: ...
    moco_type: CCC|CAC|CAA|ACC|ACA|AAC|AAA
    leverage: the loss weight
    normalizer: (x - mean) / std
    optimizer: sgd, adam, ...
    learning_policy: learning rate schedule
    """
    def __init__(
        self, model: MoCo, device,
        moco_type: str, leverage: float,
        normalizer, optimizer, learning_policy
    ):
        self.model = model
        self.device = device
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.learning_policy = learning_policy

        self.leverage = leverage
        self.engine = getattr(self, moco_type.lower())

        self.loss = AverageMeter("Loss")
        self.progress = ProgressMeter(self.loss)
        
    def save(self, path):
        torch.save(self.model.encoder_q.state_dict(), path + "/paras.pt")

    def _ccc(self, inputs_q, inputs_k):
        labels = torch.zeros(inputs_q.size(0), dtype=torch.long).to(self.device)
        # c-c-c
        self.model.train()
        self.model.adv_train(False)
        outs_q = self.model.forward_q(self.normalizer(inputs_q))
        outs_k = self.model.forward_k(self.normalizer(inputs_k))
        logits_pos = (outs_q * outs_k).sum(dim=-1, keepdim=True)
        logits_neg = outs_q @ self.model.qc
        logits = torch.cat((logits_pos, logits_neg), dim=1)
        loss_c = cross_entropy(logits, labels, reduction="mean")
        return loss_c, outs_q, outs_k

    def ccc(self, inputs_q, inputs_k, attacker):
        loss_c, _, _ = self._ccc(inputs_q, inputs_k)
        return loss_c, loss_c

    def cac(self, inputs_q, inputs_k, attacker):
        n = inputs_q.size(0)
        inputs = torch.cat((inputs_q, inputs_k), dim=0)
        labels = torch.zeros(n, dtype=torch.long).to(self.device)
        criterion = DoubleFalse(labels)
        _, inputs_adv, _ = attacker(inputs, criterion)

        loss_c, outs_q, outs_k = self._ccc(inputs_q, inputs_k)

        # c-a-c
        self.model.adv_train(True)
        inputs_q_a, inputs_k_a = inputs_adv.split(n)
        outs_q_a = outs_q
        outs_k_a = self.model.forward_k(self.normalizer(inputs_k_a))
        logits_pos_a = (outs_q_a * outs_k_a).sum(dim=-1, keepdim=True)
        logits_neg_a = outs_q_a @ self.model.qc
        logits_a = torch.cat((logits_pos_a, logits_neg_a), dim=1)
        loss_a = cross_entropy(logits_a, labels, reduction="mean")

        return loss_c, loss_a
    
    def caa(self, inputs_q, inputs_k, attacker):
        n = inputs_q.size(0)
        inputs = torch.cat((inputs_q, inputs_k), dim=0)
        labels = torch.zeros(n, dtype=torch.long).to(self.device)
        criterion = DoubleFalse(labels)
        _, inputs_adv, _ = attacker(inputs, criterion)

        loss_c, outs_q, outs_k = self._ccc(inputs_q, inputs_k)

        # c-a-a
        self.model.adv_train(True)
        inputs_q_a, inputs_k_a = inputs_adv.split(n)
        outs_q_a = outs_q
        outs_k_a = self.model.forward_k(self.normalizer(inputs_k_a))
        logits_pos_a = (outs_q_a * outs_k_a).sum(dim=-1, keepdim=True)
        logits_neg_a = outs_q_a @ self.model.qa
        logits_a = torch.cat((logits_pos_a, logits_neg_a), dim=1)
        loss_a = cross_entropy(logits_a, labels, reduction="mean")

        return loss_c, loss_a

    def acc(self, inputs_q, inputs_k, attacker):
        n = inputs_q.size(0)
        inputs = torch.cat((inputs_q, inputs_k), dim=0)
        labels = torch.zeros(n, dtype=torch.long).to(self.device)
        criterion = DoubleFalse(labels)
        _, inputs_adv, _ = attacker(inputs, criterion)

        loss_c, outs_q, outs_k = self._ccc(inputs_q, inputs_k)

        # a-c-c
        self.model.adv_train(True)
        inputs_q_a, inputs_k_a = inputs_adv.split(n)
        outs_q_a = self.model.forward_q(self.normalizer(inputs_q_a))
        outs_k_a = outs_k
        logits_pos_a = (outs_q_a * outs_k_a).sum(dim=-1, keepdim=True)
        logits_neg_a = outs_q_a @ self.model.qc
        logits_a = torch.cat((logits_pos_a, logits_neg_a), dim=1)
        loss_a = cross_entropy(logits_a, labels, reduction="mean")

        return loss_c, loss_a
    
    def aca(self, inputs_q, inputs_k, attacker):
        n = inputs_q.size(0)
        inputs = torch.cat((inputs_q, inputs_k), dim=0)
        labels = torch.zeros(n, dtype=torch.long).to(self.device)
        criterion = DoubleFalse(labels)
        _, inputs_adv, _ = attacker(inputs, criterion)

        loss_c, outs_q, outs_k = self._ccc(inputs_q, inputs_k)

        # a-c-a
        self.model.adv_train(True)
        inputs_q_a, inputs_k_a = inputs_adv.split(n)
        outs_q_a = self.model.forward_q(self.normalizer(inputs_q_a))
        outs_k_a = outs_k
        logits_pos_a = (outs_q_a * outs_k_a).sum(dim=-1, keepdim=True)
        logits_neg_a = outs_q_a @ self.model.qa
        logits_a = torch.cat((logits_pos_a, logits_neg_a), dim=1)
        loss_a = cross_entropy(logits_a, labels, reduction="mean")

        return loss_c, loss_a

    def aca_plus(self, inputs_q, inputs_k, attacker):
        n = inputs_q.size(0)
        inputs = torch.cat((inputs_q, inputs_k), dim=0)
        labels = torch.zeros(n, dtype=torch.long).to(self.device)
        criterion = DoubleFalse(labels)
        _, inputs_adv, _ = attacker(inputs, criterion)

        loss_c, outs_q, outs_k = self._ccc(inputs_q, inputs_k)

        # a-c-a
        self.model.adv_train(True)
        inputs_q_a, inputs_k_a = inputs_adv.split(n)
        outs_q_a = self.model.forward_q(self.normalizer(inputs_q_a))
        outs_k_a = outs_k
        # for adversarial memory bank
        self.model.forward_k(self.normalizer(inputs_q_a))
        logits_pos_a = (outs_q_a * outs_k_a).sum(dim=-1, keepdim=True)
        logits_neg_a = outs_q_a @ self.model.qa
        logits_a = torch.cat((logits_pos_a, logits_neg_a), dim=1)
        loss_a = cross_entropy(logits_a, labels, reduction="mean")

        return loss_c, loss_a

    def aac(self, inputs_q, inputs_k, attacker):
        n = inputs_q.size(0)
        inputs = torch.cat((inputs_q, inputs_k), dim=0)
        labels = torch.zeros(n, dtype=torch.long).to(self.device)
        criterion = DoubleFalse(labels)
        _, inputs_adv, _ = attacker(inputs, criterion)

        loss_c, outs_q, outs_k = self._ccc(inputs_q, inputs_k)

        # a-a-c
        self.model.adv_train(True)
        inputs_q_a, inputs_k_a = inputs_adv.split(n)
        outs_q_a = self.model.forward_q(self.normalizer(inputs_q_a))
        outs_k_a = self.model.forward_k(self.normalizer(inputs_k_a))
        logits_pos_a = (outs_q_a * outs_k_a).sum(dim=-1, keepdim=True)
        logits_neg_a = outs_q_a @ self.model.qc
        logits_a = torch.cat((logits_pos_a, logits_neg_a), dim=1)
        loss_a = cross_entropy(logits_a, labels, reduction="mean")

        return loss_c, loss_a

    def aaa(self, inputs_q, inputs_k, attacker):
        n = inputs_q.size(0)
        inputs = torch.cat((inputs_q, inputs_k), dim=0)
        labels = torch.zeros(n, dtype=torch.long).to(self.device)
        criterion = DoubleFalse(labels)
        _, inputs_adv, _ = attacker(inputs, criterion)

        loss_c, outs_q, outs_k = self._ccc(inputs_q, inputs_k)

        # a-a-a
        self.model.adv_train(True)
        inputs_q_a, inputs_k_a = inputs_adv.split(n)
        outs_q_a = self.model.forward_q(self.normalizer(inputs_q_a))
        outs_k_a = self.model.forward_k(self.normalizer(inputs_k_a))
        logits_pos_a = (outs_q_a * outs_k_a).sum(dim=-1, keepdim=True)
        logits_neg_a = outs_q_a @ self.model.qa
        logits_a = torch.cat((logits_pos_a, logits_neg_a), dim=1)
        loss_a = cross_entropy(logits_a, labels, reduction="mean")

        return loss_c, loss_a

    def train(self, trainloader, attacker, epoch=8888):
        self.progress.step() # reset the meter
        for inputs_q, inputs_k, _ in trainloader:
            inputs_q = inputs_q.to(self.device)
            inputs_k = inputs_k.to(self.device)

            loss_c, loss_a = self.engine(inputs_q, inputs_k, attacker)
            loss = self.leverage * loss_c + (1 - self.leverage) * loss_a
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.model.momentum_update_key_encoder() # update the key encoder

            self.loss.update(loss.item(), n=inputs_q.size(0), mode="mean")

        self.progress.display(epoch=epoch) 
        self.learning_policy.step(epoch=epoch) # update learning rate
        return self.loss.avg



class LinearCoach:

    def __init__(
        self, model, device,
        normalizer, optimizer, 
        learning_policy   
    ):
        self.model = model
        self.device = device
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.learning_policy = learning_policy
        self.loss = AverageMeter("Loss")
        self.acc = AverageMeter("Acc.")
        self.progress = ProgressMeter(self.loss, self.acc)

    def save(self, path):
        torch.save(self.model.state_dict(), path + "/paras.pt")

    
    def train(self, trainlaoder, *, epoch=8888, finetune=False, bn_adv=True):
        self.progress.step() # reset the meter
        self.model.train(finetune)
        self.model.adv_train(bn_adv)
        for inputs, labels in trainlaoder:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outs = self.model(self.normalizer(inputs))
            loss = cross_entropy(outs, labels, reduction="mean")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc_count = (outs.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.acc.update(acc_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step()

        return self.loss.avg

    def adv_train(self, trainloader, attacker, *, epoch=8888, finetune=False, bn_adv=True):
        self.progress.step()
        self.model.adv_train(bn_adv)
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            _, inputs_adv, _ = attacker(inputs, labels)

            self.model.train(finetune)
            outs = self.model(self.normalizer(inputs_adv))
            loss = cross_entropy(outs, labels, reduction="mean")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc_count = (outs.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.acc.update(acc_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step()

        return self.loss.avg

    def trades(self, trainloader, attacker, *, leverage=6., epoch=8888, finetune=False, bn_adv=True):
        self.progress.step()
        self.model.adv_train(bn_adv)
        for inputs, labels in trainloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                self.model.eval()
                logits = self.model(self.normalizer(inputs)).detach()
            criterion = LogitsAllFalse(logits) # perturbed by kl loss
            _, inputs_adv, _ = attacker(inputs, criterion)
            
            self.model.train(finetune)
            logits_clean = self.model(self.normalizer(inputs))
            logits_adv = self.model(self.normalizer(inputs_adv))
            loss_clean = cross_entropy(logits_clean, labels)
            loss_adv = kl_divergence(logits_adv, logits_clean)
            loss = loss_clean + leverage * loss_adv

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc_count = (logits_adv.argmax(-1) == labels).sum().item()
            self.loss.update(loss.item(), inputs.size(0), mode="mean")
            self.acc.update(acc_count, inputs.size(0), mode="sum")

        self.progress.display(epoch=epoch)
        self.learning_policy.step()

        return self.loss.avg

            

class Adversary:
    """
    Adversary is mainly based on foolbox, especially pytorchmodel.
    model: Make sure that the model's output is the logits or the attack is adapted.
    attacker: the attack implemented by foolbox or a similar one
    device: ...
    bounds: typically [0, 1]
    preprocessing: including mean, std, which is similar to normalizer
    criterion: typically given the labels and consequently it is Misclassification, 
            other critera could be given to carry target attack or black attack.
    """
    def __init__(
        self, model, attacker, device,
        bounds, preprocessing, epsilon
    ):
        model.eval()
        self.fmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device
        )
        self.model = model
        self.device = device
        self.epsilon = epsilon
        self.attacker = attacker 

    def attack(self, inputs, criterion, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        self.model.eval() # make sure in evaluating mode ...
        return self.attacker(self.fmodel, inputs, criterion, epsilons=epsilon)

    @torch.no_grad()
    def accuracy(self, inputs, labels):
        inputs_, labels_ = ep.astensors(inputs, labels)
        del inputs, labels

        self.model.eval() # make sure in evaluating mode ...
        predictions = self.fmodel(inputs_).argmax(axis=-1)
        accuracy = (predictions == labels_)
        return accuracy.sum().item()

    def success(self, inputs, criterion, epsilon=None):
        _, _, is_adv = self.attack(inputs, criterion, epsilon)
        return is_adv.sum().item()

    def evaluate(self, dataloader, epsilon=None, bn_adv=True):
        datasize = len(dataloader.dataset)
        running_accuracy = 0
        running_success = 0
        self.model.adv_train(bn_adv)
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            running_accuracy += self.accuracy(inputs, labels)
            running_success += self.success(inputs, labels, epsilon)
        return running_accuracy / datasize, running_success / datasize

    @enter_attack_exit
    def __call__(self, inputs, criterion, *, epsilon=None):
        return self.attack(inputs, criterion, epsilon)



class FBDefense:
    def __init__(self, model, device, bounds, preprocessing):
        self.rmodel = fb.PyTorchModel(
            model,
            bounds=bounds,
            preprocessing=preprocessing,
            device=device            
        )

        self.model = model

    @torch.no_grad()
    def query(self, inputs):
        self.model.eval()
        return self.rmodel(inputs)

    def __call__(self, inputs):
        return self.query(inputs)