
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
