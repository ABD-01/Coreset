import argparse
import logging
import pathlib

import matplotlib.pyplot as plt

# from train import test

plt.style.use("ggplot")
import gc
from pprint import pformat

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from functorch import grad, make_functional_with_buffers, vmap
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from tqdm import trange
from tqdm import tqdm

from utils.common import (
    AlexNet,
    DatasetwithIndices,
    create_config,
    get_dataset,
    get_dataset_with_indices,
    get_logger,
    get_model,
    get_optimizer,
    get_parser,
    get_test_dataset,
    get_train_dataset,
    seed_everything,
)


# def get_mean_gradients(model, loader, use_all_params=False):
#     num_params = len(
#         list(model.parameters() if use_all_params else model.fc.parameters())
#     )
#     mean_gradients = [None for i in range(num_params)]
#     num_iter = len(loader)
#     progress_bar = tqdm(
#         loader, total=num_iter, desc="Mean Gradients", leave=False, position=2
#     )
#     for batch in progress_bar:
#         images, labels, _ = batch
#         torch.cuda.empty_cache()
#         images, labels = images.to(device), labels.to(device)
#         output = model(images)
#         gradient = torch.autograd.grad(
#             F.nll_loss(output, labels),
#             model.parameters() if use_all_params else model.fc.parameters(),
#         )
#         if mean_gradients[0] is not None:
#             for j in range(num_params):
#                 mean_gradients[j] += gradient[j].detach()  # .cpu().numpy()
#         else:
#             for j in range(len(gradient)):
#                 mean_gradients[j] = gradient[j].detach()  # .cpu().numpy()

#         for j in range(len(gradient)):
#             mean_gradients[j] /= num_iter

#     return mean_gradients


# def get_similarities(model, dataset, batch_size, mean_gradients, use_all_params=False):
#     slmodel, params, buffers = make_functional_with_buffers(
#         model if use_all_params else model.fc
#     )
#     loader = DataLoader(
#         dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True
#     )

#     def loss_function(params, buffers, x, y):
#         x = x.unsqueeze(0)
#         y = y.unsqueeze(0)
#         preds = slmodel(params, buffers, x)
#         return F.nll_loss(preds, y)

#     batched_loss = vmap(
#         grad(loss_function),
#         (None, None, 0, 0),
#     )

#     similarities = []
#     img_indices = []
#     progress_bar = tqdm(
#         enumerate(loader),
#         total=len(loader),
#         desc="Per Sample Gradient Similarity",
#         leave=False,
#         position=2,
#     )
#     for i, batch in progress_bar:
#         imgs, labels, inds = batch
#         torch.cuda.empty_cache()
#         imgs, labels, inds = imgs.to(device), labels.to(device), inds.numpy()
#         with torch.no_grad():  ### TODO: Add if else for if use_all_params
#             hidden_state = model.features(imgs)
#         gradient = batched_loss(params, buffers, hidden_state, labels)
#         gc.collect()
#         torch.cuda.empty_cache()
#         # gradient = torch.autograd.grad(F.nll_loss(model(imgs), labels), model.parameters())

#         sim = (
#             torch.stack(
#                 [
#                     F.cosine_similarity(a.view(a.shape[0], -1), b.view(1, -1))
#                     for a, b in zip(gradient, mean_gradients)
#                 ],
#                 dim=-1,
#             )
#             .sum(dim=-1)
#             .detach()
#             .cpu()
#             .numpy()
#         )
#         # sim = torch.stack(list(map(lambda, gradient, mean_gradients))).sum()
#         similarities.append(sim)
#         img_indices.append(inds)
#     return np.concatenate(similarities), np.concatenate(img_indices)


# def cosinesimilarity(a,b):
#     return np.divide(np.dot(a, b.T), (np.linalg.norm(a, axis=-1, keepdims=True) * np.linalg.norm(b, keepdims=True)))

def cosinesimilarity(a,b):
    return torch.divide(torch.mm(a, b.T), (torch.linalg.norm(a, dim=-1, keepdims=True) * torch.linalg.norm(b, keepdims=True)))

def get_sims(gradients):
    mean_gradients = gradients.mean(dim=0, keepdims=True)
    return cosinesimilarity(gradients, mean_gradients)


def get_mean_gradients(p, model, loader, criterion, optimizer):
    num_params = len(list(model.get_last_layer().parameters()))
    num_iter = len(loader)
    embedding_dim = model.get_last_layer().in_features
    sample_num = len(loader.dataset)
    gradients = torch.zeros([sample_num, p.num_classes * (embedding_dim + 1)],
                                requires_grad=False, device='cuda')
    img_indices = torch.zeros([sample_num], requires_grad=False, device='cuda')
    # gradients = np.zeros([sample_num, p.num_classes * (embedding_dim + 1)])
    # img_indices = np.zeros(sample_num)
    progress_bar = tqdm(
        loader, total=num_iter, desc="Mean Gradients", leave=False, position=2
    )
    model.eval()
    model.no_grad = True
    with model.embedding_recorder:
        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad(set_to_none=True)
            images, labels, inds = batch
            torch.cuda.empty_cache()
            images, labels = images.to(device), labels.squeeze().to(device)
            output = model(images).requires_grad_(True)
            # logger.info((output.shape, labels.shape, inds.shape))
            loss = criterion(output, labels).sum()
            batch_num = labels.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, output, retain_graph=True)[0].cpu()
                weight_parameters_grads = model.embedding_recorder.embedding.cpu().view(batch_num, 1, embedding_dim).repeat(1, p.num_classes, 1) *\
                                            bias_parameters_grads.view(batch_num, p.num_classes, 1).repeat(1, 1, embedding_dim)
                
                gradients[i*p.batch_size:min((i+1)*p.batch_size, sample_num)] = torch.cat((bias_parameters_grads, weight_parameters_grads.flatten(1)), dim=1)
            img_indices[i*p.batch_size:min((i+1)*p.batch_size, sample_num)] = inds.squeeze()

    return gradients, img_indices


def train_epoch(
    loader: torch.utils.data.DataLoader,
    model: nn.Module,
    criterion: nn.NLLLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Trains model for one epoch

    Args:
        loader (torch.utils.data.DataLoader): Dataloader
        model (nn.Module): model
        criterion (nn.NLLLoss): Loss Function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): device
    """
    model.train()
    model.no_grad = False
    # losses, accs = [], []
    for (images, labels, _) in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        # losses.append(loss.item())
        # acc = output.argmax(dim=1).eq(labels).float().mean().item()
        # accs.append(acc)
    model.eval()
    # return np.mean(losses), np.mean(accs)


# def train_loop(p, model):
#     # train_loader = DataLoader(data, p.batch_size, shuffle=True)

#     train_loader = DataLoader(get_train_dataset(p), p.batch_size, shuffle=True, num_workers=2, pin_memory=True)
#     test_data = get_test_dataset(p)
#     test_loader = DataLoader(test_data, p.batch_size)

#     criterion = nn.NLLLoss()
#     optimizer = get_optimizer(p, model)

#     losses, accs = [], []
#     for epoch in trange(p.train_epochs, leave=True):
#         model.train()
#         loss, acc = train_epoch(
#             train_loader, model, criterion, optimizer, device
#         )
#         losses.append(loss)
#         accs.append(acc)
#         gc.collect()
#         torch.cuda.empty_cache()

#     test_correct = test(test_loader, model, device)
#     test_acc = test_correct / len(test_data) * 100
#     return test_acc


def gradient_mathcing(p, data, logger):
    """Calculated mean gradient for the given dataset and find per sample similarity with mean gradients

    Args:
        p (EasyDict): Hyperparameters
        data (Dataset): Dataset
        logger

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays of shape (iter, len(dataset)) for similarities calculated for each sample for every iteration and corresponding indices
    """
    iterations = p.iter
    logger.debug(len(data))
    # assert len(data) % p.batch_size == 0, "All batches are not of same shape"

    if p.per_class:
        logger.info("Finding Mean Gradients for each class individually.")
        train_labels = torch.as_tensor(data.targets)
        cls_data = [
            Subset(data, torch.argwhere(train_labels == c))
            for c in range(p.num_classes)
        ]
        logger.debug(f"len datasets: {len(data)}")
    else:
        logger.info("Finding Mean Gradients for whole dataset at once.")

    seed_everything(p.seed)
    model = get_model(p, device)
    logger.info(
        "Model Summary\n"
        + str(summary(model, (p.channel, *p.im_size), verbose=0, device=device))
    )

    train_loader = DataLoader(
        data, p.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(p, model)

    all_similarities, all_imginds = [], []
    progressbar = trange(iterations, desc="Iterations", position=0, leave=True)
    for k in progressbar:
        # if p.with_train:
        # moving to the end of loop
        if not p.with_train:
            seed_everything(p.seed + k)
            model = get_model(p, device)
            # slmodel, params, buffers = make_functional_with_buffers(model.fc)

        if not p.per_class:
            gradients, img_indices = get_mean_gradients(p, model, train_loader, criterion, optimizer) 
            # similarities, img_indices = get_similarities(
            #     model, data, p.batch_size, mean_gradients
            # )
            similarities = get_sims(gradients).cpu().numpy()
            img_indices = img_indices.cpu().numpy()
        elif p.per_class:
            similarities, img_indices = [], []
            progressbar2 = tqdm(cls_data, desc="Per CLass Gradient Mathcing", position=1, leave=False)
            for dataset in progressbar2:
                loader = DataLoader(
                    dataset,
                    p.batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                    # drop_last=True,
                )
                gradients, cls_all_inds = get_mean_gradients(p, model, loader, criterion, optimizer)
                # cls_all_sims, cls_all_inds = get_similarities(
                #     model, dataset, p.batch_size, mean_gradients, 
                # )
                cls_all_sims = get_sims(gradients).cpu().numpy().squeeze()
                cls_all_inds = cls_all_inds.cpu().numpy().squeeze()
                similarities.append(cls_all_sims)
                img_indices.append(cls_all_inds)
            try:
                similarities, img_indices = np.stack(similarities), np.stack(img_indices)
            except ValueError:
                similarities, img_indices = np.array(similarities, dtype=object), np.array(img_indices, dtype=object)

        all_similarities.append(similarities)
        all_imginds.append(img_indices)

        if p.with_train:
            train_epoch(train_loader, model, criterion, optimizer, device)
            gc.collect()
            torch.cuda.empty_cache()

    if p.per_class:
        logger.info(progressbar2)
    logger.info(progressbar)
    all_similarities, all_imginds = np.stack(all_similarities), np.stack(all_imginds)
    return all_similarities, all_imginds


def main(p, logger):

    # p = create_config(args.config, args)

    # logger.info("Hyperparameters\n" + pformat(vars(p)))

    global device
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU to run the program.")

    seed_everything(p.seed)

    # dataset
    train_data, _ = get_dataset(p)
    train_data = DatasetwithIndices(train_data)
    logger.info(f"Dataset\n{str(train_data)}")

    logger.info("Hyperparameters\n" + pformat(vars(p)))
    # p.num_classes = len(train_data.classes)
    # logger.debug(f"Num Classes: {p.num_classes}")

    all_similarities, all_imginds = gradient_mathcing(p, train_data, logger)
    logger.info(
        f"All similarities shape: {all_similarities.shape}, All imgindices shape: {all_imginds.shape}"
    )
    np.save(
        p.output_dir
        / f"all_similarities{'_perclass' if p.per_class else ''}{'_withtrain' if p.with_train else ''}.npy",
        all_similarities,
    )
    np.save(
        p.output_dir
        / f"all_imginds{'_perclass' if p.per_class else ''}{'_withtrain' if p.with_train else ''}.npy",
        all_imginds,
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Getting gradient similarity for each sample."
    # )
    # parser.add_argument("--config", help="Location of config file", required=True)
    # parser.add_argument("--seed", default=0, help="Seed")
    # parser.add_argument("--dataset", default="cifar100", help="Dataset to use")
    # parser.add_argument("--dataset_dir", default="./data", help="Dataset directory")
    # parser.add_argument("--topn", default=1000, type=int, help="Size of Coreset")
    # parser.add_argument(
    #     "--iter", default=100, type=int, help="Number of iterations for finding coreset"
    # )
    # parser.add_argument("-bs", "--batch_size", default=1000, help="BatchSize", type=int)
    # parser.add_argument(
    #     "--per_class",
    #     action="store_true",
    #     help="Specify whether to find Mean Gradients classwise",
    # )
    # parser.add_argument(
    #     "--with_train",
    #     action="store_true",
    #     help="No. of epochs to train before finding Gmean",
    # )
    # parser.add_argument(
    #     "--temp",
    #     action="store_true",
    #     help="Specify whether to use temp folder",
    # )
    # parser.add_argument(
    #     "--use_all_params",
    #     help="Specify if all model parameters' gradients to be used. Defaults: (FC layers only)",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--resume", default=None, help="path to checkpoint from where to resume"
    # )

    parser = get_parser()

    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir = pathlib.Path(args.dataset.lower()) / args.output_dir
    else:
        args.output_dir = pathlib.Path(args.dataset.lower())
    if args.pretrained:
        args.output_dir = args.output_dir / "pretrained"
    if args.temp:
        args.output_dir = args.output_dir / f"temp"
    args.logdir = args.output_dir / "logs"
    args.logdir.mkdir(parents=True, exist_ok=True)

    # temporary fix
    args.class_balanced = None
    args.augment = None
    logger = get_logger(args, "gradmatch")
    try:
        main(args, logger)
    except Exception:
        logger.exception("A Error Occurred")
