#!/usr/bin/env python

import torch
import torchvision.transforms as T
import argparse
from src.loadopts import *


METHOD = "CIFAR-C"


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str,
                help="cifar10:CIFAR-10-C, cifar100:CIFAR-100-C")
parser.add_argument("info_path", type=str)
parser.add_argument("--bn_adv", action="store_false", default=True)
parser.add_argument("-ct","--corruption_type", choices=(
    "brightness", "defocus_blur", "fog", "gaussian_blur", "glass_blur", "jpeg_compression",
    "motion_blur", "saturate", "snow", "speckle_noise", "contrast", "elastic_transform", "frost",
    "gaussian_noise", "impulse_noise", "pixelate", "shot_noise", "spatter", "zoom_blur"
), default="brightness")
parser.add_argument("-b", "--batch_size", type=int, default=128)
parser.add_argument("--progress", action="store_true", default=False, 
                help="show the progress if true")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="eval")
opts = parser.parse_args()


def load_cfg():
    from src.dict2obj import Config
    from src.base import Adversary
    from src.utils import gpu, load, set_seed
    from src.datasets import CIFAR10C, CIFAR100C, SingleSet
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
    cfg["model"] = Wrapper(arch=arch, fc=fc)
    cfg["device"] = gpu(cfg.model)
    cfg.model.eval()
    cfg.model.adv_train(opts.bn_adv)
    print(f"bn_adv: {opts.bn_adv}")
    load( # load the state dict
        model=cfg.model, 
        filename=opts.info_path + "/paras.pt", 
        device=cfg.device, strict=True
    )

    # load the testset
    if opts.dataset == "cifar10":
        dataset = CIFAR10C(corruption_type=opts.corruption_type)
    elif opts.dataset == "cifar100":
        dataset = CIFAR100C(corruption_type=opts.corruption_type)
    else:
        raise NotImplementedError(f"Supported: CIFAR-10|100-C")
    print(f"==================Corruption Type: {opts.corruption_type}==================")
    dataset = SingleSet(
        dataset=dataset,
        transform=T.ToTensor()
    )
    cfg["dataloader"] = load_dataloader(
        dataset=dataset, 
        batch_size=opts.batch_size, 
        train=False,
        show_progress=opts.progress
    )
    cfg["normalizer"] = load_normalizer(opts.dataset)

    # generate the log path
    _, log_path = generate_path(METHOD, opts.dataset, 
                        opts.model, opts.description)
    return cfg, log_path

def main(model, normalizer, device, dataloader):
    from src.utils import distance_lp, AverageMeter
    
    acc = AverageMeter("ACC")
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
      
        with torch.no_grad():
            logits = model(normalizer(inputs))
        preds = logits.argmax(-1) == labels

        acc.update(preds.sum().item(), n=inputs.size(0), mode="sum")
    
    print(acc)


if __name__ == "__main__":
    from src.utils import mkdirs, readme
    cfg, log_path = load_cfg()
    mkdirs(log_path)
    readme(log_path, opts, mode="a")

    main(**cfg)












