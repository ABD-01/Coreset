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
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from tqdm import trange
from tqdm.auto import tqdm

from utils import AlexNet, create_config, get_dataset_with_indices, get_logger


def get_mean_gradients(model, loader, use_all_params=False):
    num_params = len(
        list(model.parameters() if use_all_params else model.fc.parameters())
    )
    mean_gradients = [None for i in range(num_params)]
    num_iter = len(loader)
    progress_bar = tqdm(loader, total=num_iter, desc="Mean Gradients", leave=False)
    for batch in progress_bar:
        images, labels, _ = batch
        torch.cuda.empty_cache()
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        gradient = torch.autograd.grad(
            F.nll_loss(output, labels),
            model.parameters() if use_all_params else model.fc.parameters(),
        )
        if mean_gradients[0] is not None:
            for j in range(num_params):
                mean_gradients[j] += gradient[j].detach()  # .cpu().numpy()
        else:
            for j in range(len(gradient)):
                mean_gradients[j] = gradient[j].detach()  # .cpu().numpy()

        for j in range(len(gradient)):
            mean_gradients[j] /= num_iter

    return mean_gradients


def get_similarities(model, dataset, batch_size, mean_gradients, use_all_params=False):
    slmodel, params, buffers = make_functional_with_buffers(
        model if use_all_params else model.fc
    )
    loader = DataLoader(
        dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    def loss_function(params, buffers, x, y):
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        preds = slmodel(params, buffers, x)
        return F.nll_loss(preds, y)

    batched_loss = vmap(
        grad(loss_function),
        (None, None, 0, 0),
    )

    similarities = []
    img_indices = []
    progress_bar = tqdm(
        enumerate(loader),
        total=len(loader),
        desc="Per Sample Gradient Similarity",
        leave=False,
    )
    for i, batch in progress_bar:
        imgs, labels, inds = batch
        torch.cuda.empty_cache()
        imgs, labels, inds = imgs.to(device), labels.to(device), inds.numpy()
        with torch.no_grad():  ### TODO: Add if else for if use_all_params
            hidden_state = model.features(imgs)
        gradient = batched_loss(params, buffers, hidden_state, labels)
        gc.collect()
        torch.cuda.empty_cache()
        # gradient = torch.autograd.grad(F.nll_loss(model(imgs), labels), model.parameters())

        sim = (
            torch.stack(
                [
                    F.cosine_similarity(a.view(a.shape[0], -1), b.view(1, -1))
                    for a, b in zip(gradient, mean_gradients)
                ],
                dim=-1,
            )
            .sum(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )
        # sim = torch.stack(list(map(lambda, gradient, mean_gradients))).sum()
        similarities.append(sim)
        img_indices.append(inds)
    return np.concatenate(similarities), np.concatenate(img_indices)


def gradient_mathcing(p, train_data)->tuple[np.ndarray, np.ndarray]:
    """Calculated mean gradient for the given dataset and find per sample similarity with mean gradients

    Args:
        p (EasyDict): Hyperparameters
        train_data (Dataset): Dataset

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of shape (iter, len(dataset)) for similarities calculated for each sample for every iteration and corresponding indices
    """
    iterations = p.iter
    all_similarities, all_imginds = [], []
    for k in trange(iterations, desc="Iterations"):
        torch.manual_seed(k)
        torch.cuda.manual_seed(k)
        loader = DataLoader(
            train_data, p.batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        model = AlexNet(p.num_classes, False).to(device)
        # slmodel, params, buffers = make_functional_with_buffers(model.fc)
        mean_gradients = get_mean_gradients(model, loader, p.use_all_params)
        similarities, img_indices = get_similarities(
            model, train_data, p.batch_size, mean_gradients, p.use_all_params
        )

        all_similarities.append(similarities)
        all_imginds.append(img_indices)

    all_similarities, all_imginds = np.stack(all_similarities), np.stack(all_imginds)
    return all_similarities, all_imginds


def main(p, logger):

    # p = create_config(args.config, args)

    logger.info("Hyperparameters\n" + pformat(vars(p)))

    global device
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU to run the program.")

    # dataset
    train_data = get_dataset_with_indices(p)
    p.num_classes = len(train_data.classes)
    logger.debug(f"Num Classes: {p.num_classes}")

    if p.per_class:
        logger.info("Finding Mean Gradients for each class individually.")
        train_labels = torch.as_tensor(train_data.targets)
        datasets = [Subset(train_data, torch.argwhere(train_labels==c)) for c in train_data.dataset.class_to_idx.values()]
        logger.debug(f"len datasets: {len(datasets)}")
        all_similarities, all_imginds = [], []
        for dataset in tqdm(datasets, desc="Per CLass Gradient Mathcing"):
            cls_all_sims, cls_all_inds = gradient_mathcing(p, dataset)
            all_similarities.append(cls_all_sims)
            all_imginds.append(cls_all_inds)
        all_similarities, all_imginds = np.stack(all_similarities), np.stack(all_imginds)
        logger.debug(f"All similarities shape: {all_similarities.shape}, All imgindices shape: {all_imginds.shape}")
        np.save(p.output_dir / "all_similarities_perclass.npy", all_similarities)
        np.save(p.output_dir / "all_imginds_perclass.npy", all_imginds)
    else:
        logger.info("Finding Mean Gradients for whole dataset at once.")
        all_similarities, all_imginds = gradient_mathcing(p, train_data) 
        logger.debug(f"All similarities shape: {all_similarities.shape}, All imgindices shape: {all_imginds.shape}")
        np.save(p.output_dir / "all_similarities.npy", all_similarities)
        np.save(p.output_dir / "all_imginds.npy", all_imginds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Getting gradient similarity for each sample."
    )
    # parser.add_argument("--config", help="Location of config file", required=True)
    parser.add_argument("--seed", default=0, help="Seed")
    parser.add_argument("--dataset", default="cifar100", help="Dataset to use")
    parser.add_argument("--dataset_dir", default="./data", help="Dataset directory")
    parser.add_argument("--topn", default=1000, type=int, help="Size of Coreset")
    parser.add_argument(
        "--iter", default=100, type=int, help="Number of iterations for finding coreset"
    )
    parser.add_argument("-bs", "--batch_size", default=1000, help="BatchSize", type=int)
    parser.add_argument("--per_class", action="store_true", help="Specify whether to find Mean Gradients classwise")
    parser.add_argument(
        "--use_all_params",
        help="Specify if all model parameters' gradients to be used. Defaults: (FC layers only)",
        action="store_true",
    )
    parser.add_argument(
        "--resume", default=None, help="path to checkpoint from where to resume"
    )

    args = parser.parse_args()
    args.output_dir = pathlib.Path(args.dataset)
    args.logdir = pathlib.Path(args.dataset) / "logs"
    args.logdir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(args, "gradmatch")
    try:
        main(args, logger)
    except Exception:
        logger.exception("A Error Occurred")
