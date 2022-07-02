import logging

import matplotlib.pyplot as plt

plt.style.use("ggplot")

import numpy as np
from tqdm.auto import tqdm


def get_best_inds(
    topn: int, all_similarities: np.ndarray, all_imginds: np.ndarray
) -> np.ndarray:
    """Return n samples having maximum gradient similarity

    Args:
        topn (int): no. of best samples
        all_similarities (np.ndarray): Array of shape (iter, len(dataset)) for similarities calculated for each sample for every iteration
        all_imginds (np.ndarray): Array of shape (iter, len(dataset)) of corresponding indicies of similarities array

    Returns:
        np.ndarray: indices for images in the coreset
    """
    good_inds = []
    for (sims, inds) in tqdm(zip(all_similarities, all_imginds)):
        ind = np.argpartition(-sims, topn)[:topn]
        good_inds.append(inds[ind])
    good_inds = np.concatenate(good_inds)
    values, counts = np.unique(good_inds, return_counts=True)
    # ref:https://stackoverflow.com/a/28736715/13730689
    best_inds = np.argpartition(-counts, kth=topn)[:topn]
    return best_inds


def get_cls_balanced_best_inds(
    topn: int,
    num_classes: int,
    labels: np.ndarray,
    all_similarities: np.ndarray,
    all_imginds: np.ndarray,
) -> np.ndarray:
    """Return n samples having maximum classwise gradient similarity

    Args:
        topn (int): no. of best samples
        num_classes (int): no. of classes in the dataset
        labels (np.ndarray): true labels of the dataset
        all_similarities (np.ndarray): Array of shape (iter, len(dataset)) for similarities calculated for each sample for every iteration
        all_imginds (np.ndarray): Array of shape (iter, len(dataset)) of corresponding indicies of similarities array

    Returns:
        np.ndarray: indices for images in the coreset
    """
    topn_per_class = topn // num_classes
    cls_good_inds = [[] for i in range(num_classes)]
    for (sims, inds) in tqdm(zip(all_similarities, all_imginds)):
        shuffled_labels = labels[inds]
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
        _, counts = np.unique(cls_good_ind, return_counts=True)
        inds = np.argpartition(-counts, kth=topn_per_class)[:topn_per_class]
        best_inds.append(cls_good_ind[inds])
    best_inds = np.concatenate(best_inds)
    return best_inds


def plot_distribution(topn: int, best_labels: np.ndarray, classes: list, path) -> None:
    """Plots distirbution of classes in sampled coreset

    Args:
        topn (int): no. of best samples
        best_labels (np.ndarray): true labels for coreset
        classes (list): classes present in the dataset
        path (pathlib.Path): directory to save plots
    """
    width = max(5, len(classes) * 0.5)
    height = max(5, width // 5)
    fig = plt.figure(figsize=(width, height))
    unique_and_counts = np.unique(best_labels, return_counts=True)
    plt.bar(*unique_and_counts)
    plt.xticks(unique_and_counts[0], classes, rotation=45)
    plt.xlabel("Classes")
    plt.ylabel("Number of occurance")
    plt.title(f"Distribution of classes in selected {topn}")
    plt.grid(linestyle="--")
    for i, v in enumerate(unique_and_counts[1]):
        plt.text(i - 0.2, v + 1, str(v))
    plt.savefig(path)
    # plt.show()


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
            logging.info(f"Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                logging.info("Early stopping")
                self.early_stop = True

    # ref : https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/


def plot_learning_curves(
    losses: list, accs: list, val_losses: list, val_accs: list, topn: int, path
) -> None:
    """Plots Learning Curves

    Args:
        losses (list): Train Losses
        accs (list): Train Accuracies
        val_losses (list): Validation Losses
        val_accs (list): Validation Accuracies
        topn (int): no. of best samples
        path (pathlib.Path): directory to save plots
    """
    fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 5 * 2))
    ax1.plot(losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title(f"LossCurve_greedy{topn}")
    ax2.plot(accs, label="Train Acc")
    ax2.plot(val_accs, label="Val Acc")
    ax2.set_title(f"AccCurve_greedy{topn}")
    ax1.legend()
    ax2.legend()
    plt.savefig(path)
    plt.show()
