import argparse
import pathlib

import matplotlib.pyplot as plt

plt.style.use("ggplot")
import gc
from pprint import pformat

import numpy as np
import torch
import torch.nn.functional as F
from functorch import grad, make_functional_with_buffers, vmap
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import trange
from tqdm.auto import tqdm

from utils import AlexNet, create_config, get_dataset_with_indices, get_logger



@grad
def loss_gradient(slmodel, params, buffers, x, y):
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    preds = slmodel(params, buffers, x)
    return F.nll_loss(preds, y)


batched_loss = vmap(
    loss_gradient,
    (None, None, None, 0, 0),
)


def get_best_inds(topn, all_similarities, all_imginds):
    good_inds = []
    for (sims, inds) in tqdm(zip(all_similarities, all_imginds)):
        ind = np.argpartition(-sims, topn)[:topn]
        good_inds.append(inds[ind])
    good_inds = np.concatenate(good_inds)
    values, counts = np.unique(good_inds, return_counts=True)
    # ref:https://stackoverflow.com/a/28736715/13730689
    best_inds = np.argpartition(-counts, kth=topn)[:topn]
    return best_inds

def get_cls_balanced_best_inds(topn_per_class, all_similarities, all_imginds):
    good_inds = []
    cls_good_inds = [[] for i in range(num_classes)]
    for (sims, inds) in tqdm(zip(all_similarities, all_imginds)):
        shuffled_labels = train_labels[inds]
        for i in range(num_classes):
            cls_mask = np.where(shuffled_labels == i)[0]
            cls_sims = sims[cls_mask]
            cls_inds = inds[cls_mask]

            ind = np.argpartition(-cls_sims, topn_per_class)[:topn_per_class]
            good_ind = cls_inds[ind]
            cls_good_inds[i].append(good_ind)

    cls_good_inds = [np.concatenate(x) for x in cls_good_inds]

    best_inds = []
    for cls_good_ind in cls_good_inds:
        values, counts = np.unique(cls_good_ind, return_counts=True)
        inds = np.argpartition(-counts, kth=topn_per_class)[:topn_per_class]
        best_inds.append(cls_good_ind[inds])
    best_inds = np.concatenate(best_inds)
    return best_inds


def main(args):

    p = create_config(args.config, args)
    logger = get_logger(p)

    logger.info("Hyperparameters\n" + pformat(p))

    global device
    if args.use_gpu and torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU to run the program.")

    # dataset
    train_data = get_dataset_with_indices(p)
    num_classes = len(train_data.classes)

    # model
    model = AlexNet(output_dim=num_classes, dropout=False).to(device)
    logger.info(
        "Model Summary\n"
        + str(summary(model, train_data[1][0].shape, verbose=0, device=device))
    )

    all_similarities = np.load(p.output_dir / f"all_similarities_{p.topn}.npy")
    all_imginds = np.save(p.output_dir / f"all_imginds_{p.topn}.npy")
    logger.debug(all_similarities.shape, all_imginds.shape)

    if p.class_balanced:
        raise NotImplemented
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Getting gradient similarity for each sample."
    )
    parser.add_argument("--config", help="Location of config file", required=True)
    parser.add_argument("--seed", default=0, help="Seed")
    parser.add_argument("--use_gpu", help="Specify to run on GPU", action="store_true")
    parser.add_argument("--dataset", default="cifar100", help="Dataset Location")
    parser.add_argument("--topn", default=1000, help="Size of Coreset")
    parser.add_argument(
        "--iter", default=100, help="Number of iterations for finding coreset"
    )
    parser.add_argument("--class_balanced", help="Specify to use class balanced distribution for training", action="store_true")
    parser.add_argument("-bs", "--batch_size", default=1000, help="BatchSize", type=int)
    parser.add_argument(
        "--use_all_params",
        help="Specify if all model parameters' gradients to be used. Defaults: (FC layers only)",
        action="store_true",
    )
    parser.add_argument(
        "--resume", default=None, help="path to checkpoint from where to resume"
    )
    parser.add_argument("--wandb", default=False, type=bool, help="Log using wandb")

    args = parser.parse_args()
    args.output_dir = pathlib.Path(args.dataset)
    args.logdir = pathlib.Path(args.dataset) / "logs"
    args.logdir.mkdir(parents=True, exist_ok=True)
    main(args)
