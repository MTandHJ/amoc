#!/usr/bin/env python

import torch
import argparse
import os
from src.loadopts import *




METHOD = "WhiteBox"
FMT = "{description}={bn_adv}={attack}-{stepsize}-{steps}" \
        "={epsilon_min}-{epsilon_max}-{epsilon_times}"


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("info_path", type=str)
parser.add_argument("--filename", type=str, default="paras.pt")
parser.add_argument("--bn_adv", action="store_false", default=True)
parser.add_argument("--attack", type=str, default="pgd-linf")
parser.add_argument("--epsilon_min", type=float, default=8/255)
parser.add_argument("--epsilon_max", type=float, default=1.)
parser.add_argument("--epsilon_times", type=int, default=1)
parser.add_argument("--stepsize", type=float, default=0.1, 
                    help="pgd:rel_stepsize, cwl2:step_size, deepfool:overshoot, bb:lr")
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("--progress", action="store_false", default=True, 
                help="show the progress if true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="attack")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)



def load_cfg():
    from src.dict2obj import Config
    from src.base import Adversary
    from src.utils import gpu, load, set_seed
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
    model.adv_train(opts.bn_adv) # bn_adv or bn_clean
    print(f"bn_adv: {opts.bn_adv}")
    device = gpu(model)
    load( # load the state dict
        model=model, 
        filename=os.path.join(opts.info_path, opts.filename),
        device=device, strict=True
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
    normalizer = load_normalizer(opts.dataset)

    # generate the log path
    _, log_path = generate_path(METHOD, opts.dataset, 
                        opts.model, opts.description)

    # set the attack
    attack, bounds, preprocessing = load_attacks(
        attack_type=opts.attack, dataset_type=opts.dataset, 
        stepsize=opts.stepsize, steps=opts.steps
    )


    epsilons = torch.linspace(opts.epsilon_min, opts.epsilon_max, opts.epsilon_times).tolist()
    cfg['attacker'] = Adversary(
        model=model, attacker=attack, device=device, 
        bounds=bounds, preprocessing=preprocessing, epsilon=epsilons
    )

    return cfg, log_path

def main(attacker, testloader):
    from src.utils import distance_lp
    running_success = [0.] * opts.epsilon_times
    running_distance_linf = [0.] * opts.epsilon_times
    running_distance_l2 = [0.] * opts.epsilon_times
    for inputs, labels in testloader:
        inputs = inputs.to(attacker.device)
        labels = labels.to(attacker.device)

        _, clipped, is_adv = attacker(inputs, labels)
        dim_ = list(range(1, inputs.dim()))
        for epsilon in range(opts.epsilon_times):
            inputs_ = inputs[is_adv[epsilon]]
            clipped_ = clipped[epsilon][is_adv[epsilon]]

            running_success[epsilon] += is_adv[epsilon].sum().item()
            running_distance_linf[epsilon] += distance_lp(inputs_, clipped_, p=float('inf'), dim=dim_).sum().item()
            running_distance_l2[epsilon] += distance_lp(inputs_, clipped_, p=2, dim=dim_).sum().item()

    datasize = len(testloader.dataset)
    head = "-".join(map(str, (opts.attack, opts.epsilon_min, opts.epsilon_max, 
                        opts.epsilon_times, opts.stepsize, opts.steps)))
    for epsilon in range(opts.epsilon_times):
        running_distance_linf[epsilon] /= running_success[epsilon]
        running_distance_l2[epsilon] /= running_success[epsilon]
        running_success[epsilon] /= datasize

        # writter.add_scalar(head+"Success", running_success[epsilon], epsilon)
        # writter.add_scalars(
        #     head+"Distance", 
        #     {
        #         "Linf": running_distance_linf[epsilon],
        #         "L2": running_distance_l2[epsilon],
        #     },
        #     epsilon
        # )
    running_accuracy = list(map(lambda x: 1. - x, running_success))
   
    print("Accuracy: \n", running_accuracy)
    print("Distance-Linf: \n", running_distance_linf)
    print("Distance-L2: \n", running_distance_l2)
    

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import mkdirs, readme
    cfg, log_path = load_cfg()
    mkdirs(log_path)
    readme(log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=log_path, filename_suffix=METHOD)

    main(**cfg)

    writter.close()






