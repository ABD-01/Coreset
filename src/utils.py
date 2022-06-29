import logging
import pathlib
import sys
import time

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from easydict import EasyDict
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

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


def get_transform(p):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(**p["transformation_kwargs"]["normalize"]),
        ]
    )


def get_train_dataset(p):
    if p.dataset.lower() == "mnist":
        return MNIST(
            p.dataset_dir, train=True, transform=get_transform(p), download=True
        )
    elif p.dataset.lower() == "cifar10":
        return CIFAR10(
            p.dataset_dir, train=True, transform=get_transform(p), download=True
        )
    elif p.dataset.lower() == "cifar100":
        return CIFAR100(
            p.dataset_dir, train=True, transform=get_transform(p), download=True
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


def get_optimizer(p, model):
    if p.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=p.lr,
            momentum=p.optimizer_kwargs.momentum,
            weight_decay=p.optimizer_kwargs.weight_decay,
        )
    elif p.optimizer == "rms":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=p.lr,
            momentum=p.optimizer_kwargs.momentum,
            weight_decay=p.optimizer_kwargs.weight_decay,
        )
    elif p.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=p.lr)
    elif p.optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=p.lr)
    else:
        msg = f"Unknown value '{p.optimizer}' for argument optimizer"
        raise ValueError(msg)

    return optimizer


def get_scheduler(p, optimizer):
    if p.scheduler == "reduceonplateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, verbose=True, **p.scheduler_kwargs
        )
    elif p.scheduler == "onecyclelr":
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=p.scheduler_kwargs.max_lr,
            epochs=p.epochs,
            steps_per_epoch=p.scheduler_kwargs.len_loader,
            div_factor=p.scheduler_kwargs.max_lr / p.lr,
            final_div_factor=p.lr / p.scheduler_kwargs.min_lr,
            verbose=True,
        )
    elif p.scheduler == "exponentiallr":
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=p.scheduler_kwargs.gamma, verbose=True
        )
    elif p.scheduler == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=p.scheduler_kwargs.step_size,
            gamma=p.scheduler_kwargs.gamma,
            verbose=True,
        )
    else:
        msg = f"Unknown value '{p.scheduler}' for argument scheduler"
        raise ValueError(msg)

    return scheduler


def get_logger(p, script="train"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    output_file_handler = logging.FileHandler(
        pathlib.Path(p.logdir) / f"log_{script}_{get_time_str()}.txt"
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


def create_config(config_file_exp, args):

    with open(config_file_exp, "r") as stream:
        config = yaml.safe_load(stream)
        # print(yaml.dump(config))

    cfg = EasyDict(config)

    for k, v in vars(args).items():
        cfg[k] = v

    # for k, v in config.items():
    #     cfg[k] = v

    return cfg
