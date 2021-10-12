#!/usr/bin/env python


import torch
import torch.nn as nn
import argparse
from src.loadopts import *




METHOD = "LinearTRADES"
SAVE_FREQ = 10
FMT = "{description}={finetune}-{bn_adv}={learning_policy}-{optim}-{lr}={epochs}-{batch_size}={transform}"

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("--info_path", type=str, default="baseline", 
                help="If no info path is supported, normal training process will be applied.")
parser.add_argument("--finetune", action="store_true", default=False)
parser.add_argument("--bn_adv", action="store_false", default=True)

# adversarial training settings
parser.add_argument("--leverage", type=float, default=6.,
                help="the weight of kl loss")
parser.add_argument("--attack", type=str, default="pgd-kl")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--stepsize", type=float, default=0.25, 
                help="pgd:rel_stepsize")
parser.add_argument("--steps", type=int, default=10)

# basic settings
parser.add_argument("--optim", type=str, choices=("sgd", "adam"), default="sgd")
parser.add_argument("-mom", "--momentum", type=float, default=0.9,
                help="the momentum used for SGD")
parser.add_argument("-beta1", "--beta1", type=float, default=0.9,
                help="the first beta argument for Adam")
parser.add_argument("-beta2", "--beta2", type=float, default=0.999,
                help="the second beta argument for Adam")
parser.add_argument("-wd", "--weight_decay", type=float, default=5e-4,
                help="weight decay")
parser.add_argument("-lr", "--lr", "--LR", "--learning_rate", type=float, default=.1)
parser.add_argument("-lp", "--learning_policy", type=str, default="FC", 
                help="learning rate schedule defined in config.py")
parser.add_argument("--epochs", type=int, default=25,
                help="Suggestion: FC-25, TOTAL-40, AT-200, TRADES-76")
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("--transform", type=str, default='default', 
                help="the data augmentation which will be applied in training mode.")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="train")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)



def load_cfg():
    from src.base import LinearCoach
    from src.dict2obj import Config
    from src.base import Adversary
    from src.utils import gpu, load, load_checkpoint, set_seed
    from models.components import FC, Wrapper

    cfg = Config()
    set_seed(opts.seed)

    # load the model
    arch, dim_feature = load_model(opts.model)
    fc = FC(
        dim_feature=dim_feature,
        num_classes=get_num_classes(opts.dataset)
    )
    arch = arch()
    model = Wrapper(arch=arch, fc=fc)
    device = gpu(model)
    if opts.info_path == "baseline":
        print("Warning: No info path is provided and normal training process will be applied.")
        assert opts.finetune, "Try normal training but finetune is false!"
    else:
        load( # load the state dict
            model=arch, 
            filename=opts.info_path + "/paras.pt", 
            device=device, strict=True
        )

    # load the dataset
    trainset = load_dataset(
        dataset_type=opts.dataset,
        transform=opts.transform,
        train=True
    )
    cfg['trainloader'] = load_dataloader(
        dataset=trainset, 
        batch_size=opts.batch_size, 
        train=True,
        show_progress=opts.progress
    )
    testset = load_dataset(
        dataset_type=opts.dataset, 
        transform=opts.transform,
        train=False
    )
    cfg['testloader'] = load_dataloader(
        dataset=testset, 
        batch_size=opts.batch_size, 
        train=False,
        show_progress=opts.progress
    )

    normalizer = load_normalizer(dataset_type=opts.dataset)

    # If finetune is True, we will train the whole model otherwise the linear classifier only.
    if opts.finetune:
        optimizer = load_optimizer(
            model=model, optim_type=opts.optim, lr=opts.lr,
            momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
            weight_decay=opts.weight_decay
        )
        learning_policy = load_learning_policy(
            optimizer=optimizer, 
            learning_policy_type=opts.learning_policy, 
            T_max=opts.epochs
        )
    else:
        optimizer = load_optimizer(
            model=model.fc, optim_type=opts.optim, lr=opts.lr,
            momentum=opts.momentum, betas=(opts.beta1, opts.beta2),
            weight_decay=opts.weight_decay
        )
        learning_policy = load_learning_policy(
            optimizer=optimizer, 
            learning_policy_type=opts.learning_policy, 
            T_max=opts.epochs
        )
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad_(False)

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

    cfg['coach'] = LinearCoach(
        model=model, device=device,
        normalizer=normalizer, optimizer=optimizer,
        learning_policy=learning_policy
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

    # for validation
    cfg['valider'] = load_valider(
        model=model, device=device, dataset_type=opts.dataset
    )

    return cfg, log_path


def main(
    coach, attacker, valider, 
    trainloader, testloader, start_epoch,
    info_path
):
    from src.utils import save_checkpoint
    best_acc_rob = 0.
    for epoch in range(start_epoch, opts.epochs):

        if epoch % SAVE_FREQ == 0:
            save_checkpoint(info_path, coach.model, coach.optimizer, coach.learning_policy, epoch)

        valid_accuracy, valid_success = valider.evaluate(testloader, bn_adv=opts.bn_adv)
        if (1 - valid_success) > best_acc_rob:
            coach.save(info_path, "best_paras.pt")
            best_acc_rob = 1 - valid_success
        print(f"[Test]  TA: {valid_accuracy:.4f}  RA: {1-valid_success:.4f}")

        running_loss = coach.trades(
            trainloader, attacker,
            leverage=opts.leverage,
            epoch=epoch,
            finetune=opts.finetune,
            bn_adv=opts.bn_adv
        )
        writter.add_scalar("Loss", running_loss, epoch)
    
    train_accuracy, train_success = valider.evaluate(trainloader, bn_adv=opts.bn_adv)
    valid_accuracy, valid_success = valider.evaluate(testloader, bn_adv=opts.bn_adv)
    print(f"[Train] TA: {train_accuracy:.4f}  RA: {1-train_success:.4f}")
    print(f"[Test]  TA: {valid_accuracy:.4f}  RA: {1-valid_success:.4f}")
    writter.add_scalars("Accuracy", {"train": train_accuracy, "valid": valid_accuracy}, opts.epochs)
    writter.add_scalars("Success", {"train": train_success, "valid": valid_success}, opts.epochs)


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









































