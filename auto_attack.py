#!/usr/bin/env python

import torch
import argparse
import os
from src.loadopts import *
from models.base import AdversarialDefensiveModel
from autoattack import AutoAttack



METHOD = "AutoAttack"
FMT = "{description}={bn_adv}={norm}-{version}-{epsilon}"


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument("info_path", type=str)
parser.add_argument("--filename", type=str, default="paras.pt")
parser.add_argument("--bn_adv", action="store_false", default=True)

# for AA
parser.add_argument("--norm", choices=("Linf", "L2"), default="Linf")
parser.add_argument("--epsilon", type=float, default=8/255)
parser.add_argument("--version", choices=("standard", "plus"), default="standard")
parser.add_argument("-b", "--batch_size", type=int, default=128)

parser.add_argument("--seed", type=int, default=1)
parser.add_argument("-m", "--description", type=str, default="attack")
opts = parser.parse_args()
opts.description = FMT.format(**opts.__dict__)


class Defense(AdversarialDefensiveModel):
    """
    The inputs should be normalized 
    before fed into the model.
    """
    def __init__(self, model, normalizer):
        super(Defense, self).__init__()

        self.model = model
        self.normalizer = normalizer

    def forward(self, inputs):
        inputs_ = self.normalizer(inputs)
        return self.model(inputs_)


def load_cfg():
    from src.dict2obj import Config
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
    model.eval()
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
    data = []
    targets = []
    for i in range(len(testset)):
        img, label = testset[i]
        data.append(img)
        targets.append(label)
    
    cfg['data'] = torch.stack(data)
    cfg['targets'] = torch.tensor(targets, dtype=torch.long)
    normalizer = load_normalizer(opts.dataset)

    # generate the log path
    _, log_path = generate_path(METHOD, opts.dataset, 
                        opts.model, opts.description)

    cfg['attacker'] = AutoAttack(
        Defense(model, normalizer),
        norm=opts.norm,
        eps=opts.epsilon,
        version=opts.version,
        device=device
    )

    return cfg, log_path


def main(attacker, data, targets):
    attacker.run_standard_evaluation(data, targets, bs=opts.batch_size)



if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from src.utils import mkdirs, readme
    cfg, log_path = load_cfg()
    mkdirs(log_path)
    readme(log_path, opts, mode="a")
    writter = SummaryWriter(log_dir=log_path, filename_suffix=METHOD)

    main(**cfg)

    writter.close()


