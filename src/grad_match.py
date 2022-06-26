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
            model.parameters() if use_all_params else model.classifier.parameters(),
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


def get_similarities(model, dataset, batch_size, mean_gradients, use_all_params=False):
    slmodel, params, buffers = make_functional_with_buffers(
        model if use_all_params else model.fc
    )
    loader = DataLoader(
        dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    similarities = []
    img_indices = []
    progress_bar = tqdm(
        enumerate(loader),
        total=len(dataset) // batch_size,
        desc="Per Sample Gradient Similarity",
        leave=False,
    )
    for i, batch in progress_bar:
        imgs, labels, inds = batch
        torch.cuda.empty_cache()
        imgs, labels, inds = imgs.to(device), labels.to(device), inds.numpy()
        with torch.no_grad():  ### TODO: Add if else for if use_all_params
            hidden_state = model.features(imgs)
        gradient = batched_loss(slmodel, params, buffers, hidden_state, labels)
        gc.collect()
        torch.cuda.empty_cache()
        # gradient = torch.autograd.grad(F.nll_loss(model(imgs), labels), model.parameters())

        sim = (
            torch.stack(
                [
                    F.cosine_similarity(a.view(batch_size, -1), b.view(1, -1))
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

    iterations = p.iter
    all_similarities, all_imginds = [], []
    for k in trange(iterations, desc="Iterations"):
        torch.manual_seed(k)
        torch.cuda.manual_seed(k)
        loader = DataLoader(
            train_data, p.batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        model = AlexNet(100, False).to(device)
        # slmodel, params, buffers = make_functional_with_buffers(model.classifier)
        mean_gradients = get_mean_gradients(model, loader, p.use_all_params)
        similarities, img_indices = get_similarities(
            model, train_data, p.batch_size, mean_gradients, p.use_all_params
        )

        all_similarities.append(similarities)
        all_imginds.append(img_indices)

    np.stack(all_similarities).shape, np.stack(all_imginds).shape
    np.save(p.output_dir / f"all_similarities_{p.topn}.npy", all_similarities)
    np.save(p.output_dir / f"all_imginds_{p.topn}.npy", all_imginds)


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
