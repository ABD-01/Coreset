import logging
import os
import argparse
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import yaml
from easydict import EasyDict
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

# add import for nets directory just outside the src direcory
sys.path.append(str(pathlib.Path(__file__).parent.parent))
import nets
import datasets


def seed_everything(seed: int):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # compared to manual seed, it works for all GPUS
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# def dataset_with_indices(cls):

#     def __getitem__(self, index):
#         data, target = cls.__getitem__(self, index)
#         return data, target, index

#     return type(
#         cls.__name__,
#         (cls,),
#         {
#             "__getitem__": __getitem__,
#         },
#     )
# https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19


def get_transform(p, aug=False):

    normalize = T.Compose(
        [T.ToTensor(), T.Normalize(**p["transformation_kwargs"]["normalize"])]
    )

    augment = T.Compose(
        [
            T.RandomCrop(p.input_shape[-1], padding=4),
            T.RandomHorizontalFlip(),
            T.RandomRotation(degrees=10),
            T.RandomGrayscale(),
            # T.RandomApply([T.ColorJitter(0.5, 0.2, )]),
        ]
    )

    return T.Compose([augment, normalize]) if aug else normalize


def get_train_dataset(p, val=False):
    if p.dataset.lower() == "mnist":
        return MNIST(
            p.dataset_dir,
            train=True,
            transform=get_transform(p, p.augment and not val),
            download=True,
        )
    elif p.dataset.lower() == "cifar10":
        return CIFAR10(
            p.dataset_dir,
            train=True,
            transform=get_transform(p, p.augment and not val),
            download=True,
        )
    elif p.dataset.lower() == "cifar100":
        return CIFAR100(
            p.dataset_dir,
            train=True,
            transform=get_transform(p, p.augment and not val),
            download=True,
        )
    else:
        msg = f"Unknown value '{p.dataset}' for argument dataset"
        raise ValueError(msg)


def get_test_dataset(p):
    if p.dataset.lower() == "mnist":
        return MNIST(
            p.dataset_dir, train=False, transform=get_transform(p), download=True
        )
    elif p.dataset.lower() == "cifar10":
        return CIFAR10(
            p.dataset_dir, train=False, transform=get_transform(p), download=True
        )
    elif p.dataset.lower() == "cifar100":
        return CIFAR100(
            p.dataset_dir, train=False, transform=get_transform(p), download=True
        )
    else:
        msg = f"Unknown value '{p.dataset}' for argument dataset"
        raise ValueError(msg)


class DatasetwithIndices(Dataset):
    """Modifies the given Dataset class to return a tuple data, target, index instead of just data, target.

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, dataset):
        self.dataset = dataset
        # self.__dict__.update(self.dataset.__dict__)
        self.classes = self.dataset.classes
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return self.dataset.__len__()

    def __repr__(self) -> str:
        return "DatasetwithIndices " + self.dataset.__repr__()[8:]


def get_dataset_with_indices(p, train=True):
    dataset = get_train_dataset(p) if train else get_test_dataset(p)
    return DatasetwithIndices(dataset)


class AlexNet(nn.Module):
    def __init__(self, output_dim, dropout=True):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3, 64, 3, 2, 1
            ),  # in_channels, out_channels, kernel_size, stride, padding
            nn.MaxPool2d(2),  # kernel_size
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) if dropout else nn.Identity(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
            nn.LogSoftmax(dim=-1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity="relu",
                )
                nn.init.normal_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.features(x)
        # h = x.view(x.shape[0], -1)
        x = self.fc(h)
        return x  # , h

def get_dataset(p):
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[p.dataset](p.dataset_dir)
    p.channel, p.im_size, p.num_classes, p.class_names, p.mean, p.std = channel, im_size, num_classes, class_names, mean, std
    return dst_train, dst_test

def get_model(p, device):
    model = nets.__dict__[p.model](p.channel, p.num_classes, im_size=p.im_size).to(device)
    if device == "cpu":
        print("Using CPU.")
    elif p.gpu is not None:
        torch.cuda.set_device(p.gpu[0])
        model = nets.nets_utils.MyDataParallel(model, device_ids=p.gpu)
    elif torch.cuda.device_count() > 1:
        model = nets.nets_utils.MyDataParallel(model).cuda()

    return model

def get_optimizer(p, model):
    if p.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=p.lr,
            momentum=p.momentum,
            weight_decay=p.weight_decay,
            nesterov=p.nesterov,
        )
    elif p.optimizer == "rms":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=p.lr,
            momentum=p.momentum,
            weight_decay=p.weight_decay,
        )
    elif p.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    elif p.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    else:
        msg = f"Unknown value '{p.optimizer}' for argument optimizer"
        raise ValueError(msg)

    return optimizer


def get_scheduler(p, optimizer):
    if p.scheduler == "reduceonplateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=p.factor, patience=p.patience, verbose=True)
    elif p.scheduler == "onecyclelr":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=p.max_lr, epochs=p.epochs, steps_per_epoch=p.len_loader, div_factor=p.max_lr/p.lr, final_div_factor=p.lr/p.min_lr, verbose=False)
    elif p.scheduler == "exponentiallr":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=p.gamma, verbose=True)
    elif p.scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(optimizer,
            step_size=p.step_size,
            gamma=p.gamma,
            verbose=True,
        )
    # add  elif for cosine annealing lr
    elif p.scheduler == "cosineannealinglr":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p.T_max, eta_min=p.min_lr, verbose=False)
    elif p.scheduler == "cosineannealingwarmrestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=p.T_max, T_mult=p.T_mult, eta_min=p.min_lr, verbose=False)
    else:
        msg = f"Unknown value '{p.scheduler}' for argument scheduler"
        raise ValueError(msg)

    return scheduler


def get_logger(p, script="train"):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.getLogger("matplotlib.font_manager").disabled = True
    logging.getLogger("matplotlib.pyplot").disabled = True
    logging.getLogger("PIL").disabled = True

    suffix = (
        str(
            "_clsbalanced"
            if p.class_balanced
            else "_perclass"
            if p.per_class
            else "_baseline"
        )
        + str("_withtrain" if p.with_train else "")
        + str("_augment" if p.augment else "")
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    output_file_handler = logging.FileHandler(
        pathlib.Path(p.logdir) / f"log_{script}_{get_time_str()}{suffix}.txt"
    )
    output_file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s: %(levelname)s: %(message)s")
    stdout_handler.setFormatter(formatter)
    output_file_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(output_file_handler)
    return logger


def get_time_str():
    return time.asctime().replace(":", "-")


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            if key in ["epochs", "batch_size"]:
                getattr(namespace, self.dest)[key] = int(value)
                continue
            getattr(namespace, self.dest)[key] = float(value)


def create_config(config_file_exp, args):

    with open(config_file_exp, "r") as stream:
        config = yaml.safe_load(stream)
        # print(yaml.dump(config))

    cfg = EasyDict(config)

    for k, v in vars(args).items():
        cfg[k] = v

    cfg.update(cfg.kwargs)

    # for k, v in config.items():
    #     cfg[k] = v

    return cfg


def str_to_bool(v):
    # Handle boolean type in arguments.
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser(description="")
    # General parameters
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='AlexNet', help='model')
    parser.add_argument('--dataset_dir', type=str, default='data', help='dataset path')
    parser.add_argument('--gpu', type=int, nargs='+', default=None, help='gpu id')
    parser.add_argument('--topn', type=int, default=1000, help='Size of Coreset')
    parser.add_argument('--iter', '--num_iter', type=int, default=100, help='Number of iterations for each coreset')
    parser.add_argument('--batch_size', '-bs', type=int, default=1000, help='Batch size')
    parser.add_argument('--val_batch_size', '-vbs', type=int, default=1000, help='Validation batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--r', type=int, default=2, help='Number of workers')
    parser.add_argument('--seed', type=int, default=0, help='Seed')
    # Optimizer and Scheduler parameters
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--min_lr', type=float, default=0.0001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.01, help='weight decay (default: 0.01')
    parser.add_argument('--nesterov', type=str_to_bool, default=False, help='nesterov')
    parser.add_argument('--scheduler', type=str, default='reduceonplateau', help='scheduler')
    parser.add_argument('--patience', type=int, default=50, help='Patienece for ReduceOnPlateau')
    parser.add_argument('--factor', type=float, default=0.2, help='Factor for ReduceOnPlateau')
    parser.add_argument('--step_size', type=int, default=30, help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
    parser.add_argument('--max_lr', type=float, default=0.005, help='Max learning rate')
    parser.add_argument('--T_max', type=int, default=50, help='T_max for CosineAnnealingLR')
    parser.add_argument('--T_mult', type=int, default=1, help='T_mult for CosineAnnealingLR')
    # Early Stopping parameters
    parser.add_argument('-esp','--early_stopping_patience', type=int, default=-1, help='Early Stopping Patience (default: -1 i.e no early stopping)')
    parser.add_argument('-esd','--early_stopping_delta', type=float, default=0.001, help='Early Stopping Delta (default: 0.001)')
    parser.add_argument('-esm','--early_stopping_min_epochs', type=int, default=100, help='Minimum number of epochs for early stopping (default: 100)')
    # Misc parameters
    # params if specified true: per_class, class_balanced, with_train, augment
    parser.add_argument('--per_class', type=str_to_bool, default=False, help='Specify if per class mean gradients or not')
    parser.add_argument('--class_balanced', type=str_to_bool, default=False, help='whether to use class balanced sampling')
    parser.add_argument('--with_train', type=str_to_bool, default=False, help='With train')
    parser.add_argument('--random', type=str_to_bool, default=False, help='whether to use random sampling for coreset')
    parser.add_argument('--augment', type=str_to_bool, default=False, help='Augment')
    parser.add_argument('--temp', type=str_to_bool, default=False, help='whether to use temp folder')
    # parameters for train.py
    parser.add_argument('--train', type=str_to_bool, default=True, help='whether to train')
    parser.add_argument('--test_model', type=str, default=None, help='path to test model')

    return parser