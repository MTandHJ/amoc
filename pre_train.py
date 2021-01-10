#!/usr/bin/env python


import torch
import torch.nn as nn
import argparse
from src.loadopts import *


METHOD = "PreTrain"
SAVE_FREQ = 10
FMT = "{description}={learning_policy}-{optim}-{lr}" \
        "={moco_type}-{leverage}" \
        "={dim_mlp}-{dim_head}" \
        "={temperature}-{num_keys_clean}-{num_keys_adv}-{moco_mom}" \
        "={attack}-{epsilon:.4f}-{stepsize}-{steps}" \
        "={epochs}-{warmup_epochs}-{batch_size}={transform}"


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)

# MoCo settings
parser.add_argument("--moco_type", type=str, default="ACA")
parser.add_argument("--leverage", type=float, default=0.5)
parser.add_argument("--dim_mlp", type=int, default=512)
parser.add_argument("--dim_head", type=int, default=128)
parser.add_argument("-T", "--temperature", type=float, default=0.2)
parser.add_argument("-K1", "--num_keys_clean", type=int, default=32768)
parser.add_argument("-K2", "--num_keys_adv", type=int, default=32768)
parser.add_argument("--moco_mom", type=float, default=0.999)

# adversarial training settings
parser.add_argument("--attack", type=str, default="pgd-linf")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--stepsize", type=float, default=0.25, 
                    help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--steps", type=int, default=5)

# basic settings
parser.add_argument("--optim", type=str, choices=("sgd", "adam"), default="sgd")
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                help="the momentum used for SGD")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                help="the first beta argument for Adam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.99,
                help="the second beta argument for Adam")
parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4,
                help="weight decay")
parser.add_argument("-lr", "--lr", "--LR", "--learning_rate", type=float, default=0.1)
parser.add_argument("-lp", "--learning_policy", type=str, default="cosine", 
                help="learning rate schedule defined in config.py")
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--warmup_epochs", type=int, default=10,
                help="the number of linear warmup epochs")
parser.add_argument("-b", "--batch_size", type=int, default=256)
parser.add_argument("--transform", type=str, default='simclr')
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="pretrain")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)



def load_cfg():
    from src.base import Coach, Adversary
    from src.utils import gpu, load_checkpoint, set_seed
    from src.dict2obj import Config
    from models.components import Projector

    cfg = Config()
    set_seed(opts.seed)

    # load the model
    encoder, dim_feature = load_model(opts.model)
    projector = Projector
    model = load_moco(
        moco_type=opts.moco_type,
        base_encoder=encoder,
        base_projector=projector,
        dim_feature=dim_feature,
        dim_mlp=opts.dim_mlp,
        dim_head=opts.dim_head,
        momentum=opts.moco_mom,
        T=opts.temperature,
        K1=opts.num_keys_clean,
        K2=opts.num_keys_adv
    )
    device = gpu(model)

    # load the trainset
    trainset = load_dataset(
        dataset_type=opts.dataset,
        transform=opts.transform,
        train=True,
        double=True
    )
    cfg['trainloader'] = load_dataloader(
        dataset=trainset, 
        batch_size=opts.batch_size, 
        train=True,
        show_progress=opts.progress
    )
    normalizer = load_normalizer(dataset_type=opts.dataset)

    # load the optimizer and learning_policy
    optimizer = load_optimizer(
        model=model, optim_type=opts.optim, lr=opts.lr,
        momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
        weight_decay=opts.weight_decay, nesterov=False
    )
    learning_policy = load_learning_policy(
        optimizer=optimizer, learning_policy_type=opts.learning_policy, 
        T_max=opts.epochs-opts.warmup_epochs
    )
    warmup_lp = WarmupLP(
        learning_policy=learning_policy,
        optimizer=optimizer,
        base_lr=opts.lr,
        warmup_epochs=opts.warmup_epochs
    )

    # generate the path for logging information and saving parameters
    cfg['info_path'], log_path = generate_path(
        method=METHOD, dataset_type=opts.dataset, 
        model=opts.model, description=opts.description
    )
    if opts.resume:
        cfg['start_epoch'] = load_checkpoint(
            path=cfg.info_path, model=model, 
            optimizer=optimizer, lr_scheduler=learning_policy
        )
    else:
        cfg['start_epoch'] = 0

    # load the loss_func
    cfg['coach'] = Coach(
        model=model, device=device,
        moco_type=opts.moco_type, leverage=opts.leverage,
        normalizer=normalizer, optimizer=optimizer, learning_policy=warmup_lp
    )

    # set the attack
    attack, bounds, preprocessing = load_attacks(
        attack_type=opts.attack, dataset_type=opts.dataset, 
        stepsize=opts.stepsize, steps=opts.steps
    )

    cfg['attacker'] = Adversary(
        model=model, attacker=attack, device=device, 
        bounds=bounds, preprocessing=preprocessing, epsilon=opts.epsilon
    )

    return cfg, log_path




def main(
    coach, attacker,
    trainloader, start_epoch,
    info_path
):  
    from src.utils import save_checkpoint
    for epoch in range(start_epoch, opts.epochs):

        if epoch % SAVE_FREQ == 0:
            save_checkpoint(info_path, coach.model, coach.optimizer, coach.learning_policy, epoch)

        running_loss = coach.train(trainloader, attacker, epoch=epoch)
        writter.add_scalar("Loss", running_loss, epoch)

if __name__ ==  "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import mkdirs, readme
    cfg, log_path = load_cfg()
    mkdirs(cfg.info_path, log_path)
    readme(cfg.info_path, opts)
    readme(log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=log_path, filename_suffix=METHOD)

    main(**cfg)

    cfg['coach'].save(cfg.info_path)
    writter.close()




    
































