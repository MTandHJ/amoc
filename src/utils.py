





import torch
import os
import sys
import numpy as np
import random



class AverageMeter:

    def __init__(self, name, fmt=".6f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1, mode="mean"):
        self.val = val
        self.count += n
        if mode == "mean":
            self.sum += val * n
        elif mode == "sum":
            self.sum += val
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} Sum: {sum:{fmt}} Avg:{avg:{fmt}}"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, *meters: AverageMeter, prefix=""):
        self.meters = list(meters)
        self.prefix = prefix

    def display(self, epoch=8888):
        entries = [self.prefix + f"[Epoch: {epoch:<4d}]"]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def add(self, *meters: AverageMeter):
        self.meters += list(meters)

    def step(self):
        for meter in self.meters:
            meter.reset()

def gpu(*models):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model in models:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        else:
            model.to(device)
    return device

def mkdirs(*paths):
    for path in paths:
        try:
            os.makedirs(path)
        except FileExistsError:
            pass

def readme(path, opts, mode="w"):
    """
    opts: the argparse
    """
    import time
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S")
    filename = path + "/README.md"
    s = "- {0[0]}:  {0[1]}\n"
    info = "\n## {0}".format(time_)
    for item in opts._get_kwargs():
        info += s.format(item)
    with open(filename, mode, encoding="utf8") as fh:
        fh.write(info)

# load model's parameters
def load(model, filename, device, strict=True, except_key=None):
    """
    :param model:
    :param filename:
    :param device:
    :param except_key: drop the correspoding key module
    :return:
    """

    if str(device) =="cpu":
        state_dict = torch.load(filename, map_location="cpu")
        
    else:
        state_dict = torch.load(filename)
    if except_key is not None:
        except_keys = list(filter(lambda key: except_key in key, state_dict.keys()))
        for key in except_keys:
            try:
                del state_dict[key]
            except KeyError:
                print(f"Warning: No such key {key}.")
    model.load_state_dict(state_dict, strict=strict)
    model.eval()

# save the checkpoint
def save_checkpoint(path, model, optimizer, lr_scheduler, epoch):
    path = path + "/model-optim-lr_sch-epoch.tar"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch
        },
        path
    )

# load the checkpoint
def load_checkpoint(path, model, optimizer, lr_scheduler):
    path = path + "/model-optim-lr_sch-epoch.tar"
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    epoch = checkpoint['epoch']
    return epoch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# caculate the lp distance along the dim you need,
# dim could be tuple or list containing multi dims.
def distance_lp(x, y, p, dim=None):
    return torch.norm(x-y, p, dim=dim)
