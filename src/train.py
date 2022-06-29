import argparse
import pathlib

import matplotlib.pyplot as plt

plt.style.use("ggplot")
import gc
from pprint import pformat

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchsummary import summary

from train_utils import *
from utils import (AlexNet, create_config, get_dataset_with_indices,
                   get_logger, get_optimizer, get_test_dataset,
                   get_train_dataset)


class EarlyStopping:
    """Early stopping to stop the training when the loss does not improve after certain epochs."""

    def __init__(self, patience=10, min_delta=1e-4, threshold=0.3):
        """
        Args:
            patience (int, optional): how many epochs to wait before stopping when loss is not improving. Defaults to 10.
            min_delta (float, optional): minimum difference between new loss and old loss for new loss to be considered as an improvement. Defaults to 1e-4.
            threshold (float, optional): minimum value to be attained before the counter starts. Defaults to 0.3.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.threshold = threshold
        self.counter = 0
        self.best_acc = None
        self.early_stop = False

    def __call__(self, val_acc):
        if val_acc < self.threshold:
            return
        if self.best_acc == None:
            self.best_acc = val_acc
        elif val_acc - self.best_acc > self.min_delta:
            self.best_acc = val_acc
            # reset counter if validation acc improves
            self.counter = 0
        elif val_acc - self.best_acc < self.min_delta:
            self.counter += 1
            logger.info(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                logger.info("Early stopping")
                self.early_stop = True
# ref : https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/


@torch.inference_mode()
def test(loader, model, device):
    model.eval()
    correct = 0
    for (images, labels) in loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        acc = output.argmax(dim=1).eq(labels).sum().item()
        correct += acc
    return correct


@torch.no_grad()
def validate(loader, model, criterion, device):
    model.eval()
    val_i, val_l = next(iter(loader))
    val_i, val_l = val_i.to(device), val_l.to(device)
    model.eval()
    output = model(val_i)
    loss = criterion(output, val_l)
    acc = output.argmax(dim=1).eq(val_l).float().mean().item()
    return loss, acc


def train_epoch(loader, model, criterion, optimizer, device):
    optimizer.zero_grad(set_to_none=True)
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    output = model(images)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
    acc = output.argmax(dim=1).eq(labels).float().mean().item()
    return loss, acc


def main(args):

    p = create_config(args.config, args)

    logger.info("Hyperparameters\n" + pformat(p))

    global device
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU to run the program.")

    # dataset
    data = get_train_dataset(p)
    test_data = get_test_dataset(p)
    num_classes = len(data.classes)
    train_labels = np.array(data.targets)

    # model
    model = AlexNet(output_dim=num_classes, dropout=True).to(device)
    logger.info(
        "Model Summary\n"
        + str(summary(model, data[1][0].shape, verbose=0, device=device))
    )

    all_similarities = np.load(p.output_dir / f"all_similarities_{p.topn}.npy")
    all_imginds = np.load(p.output_dir / f"all_imginds_{p.topn}.npy")
    logger.info(
        f"all_similarities.shape: {all_similarities.shape}, all_imginds.shape: {all_imginds.shape}"
    )

    if p.class_balanced:
        best_inds = get_cls_balanced_best_inds(
            p.topn, num_classes, train_labels, all_similarities, all_imginds
        )
    else:
        best_inds = get_best_inds(p.topn, all_similarities, all_imginds)
        plot_distribution(p.topn, train_labels[best_inds], data.classes, p.output_dir)
    best_inds = torch.from_numpy(best_inds)

    val_size = best_inds.shape[0] * p.val
    sections = (best_inds.shape[0] - val_size, val_size)
    train_inds, val_inds = torch.split(torch.from_numpy(best_inds), sections)

    train_loader = DataLoader(
        Subset(data, train_inds), train_inds.shape[0], shuffle=True
    )
    val_loader = DataLoader(Subset(data, val_inds), val_inds.shape[0])
    test_loader = DataLoader(test_data, p.batch_size)

    criterion = nn.NLLLoss()
    optimizer = get_optimizer(p, model)

    early_stopping = EarlyStopping(**p.early_stopping_kwargs)
    losses, accs, val_losses, val_accs = [], [], [], []
    for epoch in trange(p.epochs):
        model.train()
        loss, acc = train_epoch(train_loader, model, criterion, optimizer, device)
        losses.append(loss.item())
        accs.append(acc)
        val_loss, val_acc = validate(val_loader, model, criterion, device)
        val_losses.append(val_loss.item())
        val_accs.append(val_acc)
        early_stopping(val_acc)
        # logger.info(f"Epoch[{epoch+1:4}] Val_Loss: {val_loss:.3f}\tVal_Acc: {val_acc:.3f}")
        gc.collect()
        torch.cuda.empty_cache()
        if epoch % 50 == 0:
            correct = test(test_loader, model, device)
            logger.info(
                f"Epoch[{epoch+1:4}] Loss: {loss.item():.2f}\tAccuracy: {acc*100 :.3f}\tVal_Loss: {val_loss:.3f}\tVal_Acc: {val_acc*100:.3f}"
            )
            logger.info(
                f"Epoch[{epoch+1:4}] Test Accuracy: {(correct / len(test_data))*100 :.3f}"
            )
        if early_stopping.early_stop:
            logger.info(f"Trained for {epoch+1} Epochs.")
            break

    plot_learning_curves(losses, accs, val_losses, val_accs, p.topn, p.output_dir)

    model.eval()
    _, train_acc = validate(train_loader, model, criterion, device)
    logger.info("Accuracy on Train Set", train_acc * 100)
    correct = test(test_loader, model)
    logger.info(correct, "correctly labeled out of", len(test_data))
    test_acc = correct / len(test_data) * 100
    logger.info("Accuracy on Test Set:", test_acc)

    torch.save(
        model.state_dict(),
        f"Greedy_Model_{p.topn}n_Epochs_{p.epochs}_Early_Stop_{epoch+1}_Test_Acc_{int(test_acc)}.pth",
    )
    logger.info("Training Complete")
    logger.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Getting gradient similarity for each sample."
    )
    parser.add_argument("--config", required=True, help="Location of config file")
    parser.add_argument("--seed", default=0, help="Seed")
    parser.add_argument(
        "--dataset", default="cifar100", required=True, help="Dataset Location"
    )
    parser.add_argument("--dataset_dir", default="./data", help="Dataset directory")
    parser.add_argument("--topn", default=1000, type=int, help="Size of Coreset")
    parser.add_argument(
        "--class_balanced",
        help="Specify to use class balanced distribution for training",
        action="store_true",
    )
    parser.add_argument("-bs", "--batch_size", default=1000, type=int, help="BatchSize")
    parser.add_argument(
        "-val",
        "--val_percent",
        default=0.1,
        type=float,
        help="Percentage[0-1] split of Validation set. (Default: 0.1)",
    )
    parser.add_argument(
        "--resume", default=None, help="path to checkpoint from where to resume"
    )
    parser.add_argument("--wandb", default=False, type=bool, help="Log using wandb")

    args = parser.parse_args()
    args.output_dir = pathlib.Path(args.dataset)
    args.logdir = pathlib.Path(args.dataset) / "logs"
    args.logdir.mkdir(parents=True, exist_ok=True)

    global logger
    logger = get_logger(args, "train")
    try:
        main(args, logger)
    except Exception:
        logger.exception("A Error Occurred")
