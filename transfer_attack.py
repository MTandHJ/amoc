#!/usr/bin/env python


"""
Transfer Attack: use the source_model to attack 
the target model...
"""

import torch
import argparse
from src.loadopts import *

METHOD = "Transfer"
FMT = "{description}={source_bn_adv}-{target_bn_adv}={attack}-{stepsize}-{steps}"


parser = argparse.ArgumentParser()
parser.add_argument("source_model", type=str)
parser.add_argument("source_path", type=str)
parser.add_argument("target_model", type=str)
parser.add_argument("target_path", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("--source_bn_adv", action="store_false", default=True)
parser.add_argument("--target_bn_adv", action="store_false", default=True)
parser.add_argument("--attack", type=str, default="pgd-linf")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--stepsize", type=float, default=0.1, 
                    help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="attack")
opts = parser.parse_args()



def load_cfg():
    from src.dict2obj import Config
    from src.base import Adversary, FBDefense
    from src.utils import gpu, load
    from models.components import FC, Wrapper
    cfg = Config()

    # load the source_model
    source_arch, dim_feature = load_model(opts.source_model)
    source_fc = FC(
        dim_feature=dim_feature,
        num_classes=get_num_classes(opts.dataset)
    )
    source_arch = source_arch()
    source_model = Wrapper(
        arch=source_arch,
        fc=source_fc
    )
    source_model.adv_train(opts.source_bn_adv)
    print(f"Source bn_adv: {opts.source_bn_adv}")
    device = gpu(source_model)
    load(
        model=source_model, 
        filename=opts.source_path + "/paras.pt", 
        device=device,
        strict=True
    )

    # load the target_model
    target_arch, dim_feature = load_model(opts.target_model)
    target_fc = FC(
        dim_feature=dim_feature,
        num_classes=get_num_classes(opts.dataset)
    )
    target_arch = target_arch()
    target_model = Wrapper(
        arch=target_arch,
        fc=target_fc
    )
    target_model.adv_train(opts.target_bn_adv)
    print(f"Target bn_adv: {opts.target_bn_adv}")
    device = gpu(target_model)
    load(
        model=target_model, 
        filename=opts.source_path + "/paras.pt", 
        device=device,
        strict=True
    )

    # load the testset
    testset = load_dataset(
        dataset_type=opts.dataset, 
        transform='None',
        train=False
    )
    cfg['testloader'] = load_dataloader(
        dataset=testset, 
        batch_size=opts.batch_size, 
        train=False,
        show_progress=opts.progress
    )

    # generate the log path
    mix_model = opts.source_model + "---" + opts.target_model
    _, log_path = generate_path(METHOD, opts.dataset,
                        mix_model, opts.description)

    # set the attack
    attack, bounds, preprocessing = load_attacks(
        attack_type=opts.attack, dataset_type=opts.dataset, 
        stepsize=opts.stepsize, steps=opts.steps
    )

    cfg['attacker'] = Adversary(
        source_model, attack, device,
        bounds, preprocessing, opts.epsilon
    )

    # set the defender ...
    cfg['defender'] = FBDefense(
        target_model, device,
        bounds, preprocessing
    )

    return cfg, log_path

def main(defender, attacker, testloader):
    from src.criteria import TransferClassification
    from src.utils import distance_lp
    running_success = 0.
    running_distance_linf = 0.
    running_distance_l2 = 0.
    for inputs, labels in testloader:
        inputs = inputs.to(attacker.device)
        labels = labels.to(attacker.device)

        criterion = TransferClassification(defender, labels)
        _, clipped, is_adv = attacker(inputs, criterion)
        inputs_ = inputs[is_adv]
        clipped_ = clipped[is_adv]

        dim_ = list(range(1, inputs.dim()))
        running_distance_linf += distance_lp(inputs_, clipped_, p=float("inf"), dim=dim_).sum().item()
        running_distance_l2 += distance_lp(inputs_, clipped_, p=2, dim=dim_).sum().item()
        running_success += is_adv.sum().item()

    running_distance_linf /= running_success
    running_distance_l2 /= running_success
    running_success /= len(testloader.dataset)

    results = "RA: {0:.6f}, Linf: {1:.6f}, L2: {2:.6f}".format(
        1-running_success, running_distance_linf, running_distance_l2
    )
    head = "-".join(map(str, (opts.attack, opts.epsilon, opts.stepsize, opts.steps)))
    writter.add_text(head, results)
    print(results)




if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import mkdirs, readme
    cfg, log_path = load_cfg()
    mkdirs(log_path)
    readme(log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=log_path, filename_suffix=METHOD)

    main(**cfg)

    writter.close()
































