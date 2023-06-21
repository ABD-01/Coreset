import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from tqdm.auto import trange

plt.style.use("ggplot")
import gc
from pprint import pformat

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchsummary import summary

from utils.train_utils import *

from utils.common import *


# def get_train_val_inds(p, best_inds: torch.Tensor):
#     """Get train and validation split for coreset

#     Args:
#         p (EasyDict): Hyperparameters
#         best_inds (torch.Tensor): indices for images in the coreset

#     Returns:
#         tuple[torch.Tensor, torch.Tensor]: indices for train and val split
#     """

#     if not (p.class_balanced or p.per_class):
#         val_size = int(best_inds.shape[0] * p.val_percent)
#         if val_size == 0:
#             return best_inds, None
#         sections = (best_inds.shape[0] - val_size, val_size)
#         train_inds, val_inds = torch.split(best_inds, sections)
#     else:
#         topn_per_class = p.topn // p.num_classes
#         val_size = int(topn_per_class * p.val_percent)
#         if val_size == 0:
#             return best_inds, None
#         val_inds = np.zeros(topn_per_class, dtype=bool)
#         val_inds[-val_size:] = True
#         val_inds = np.tile(val_inds, best_inds.shape[0] // topn_per_class)
#         train_inds = ~val_inds
#         train_inds, val_inds = best_inds[train_inds], best_inds[val_inds]
#     return train_inds, val_inds


def get_train_val_inds(len_data, best_inds):
    # return best indices and indices which are not in np.arange(len_data) as val_inds
    val_inds = np.setdiff1d(np.arange(len_data), best_inds, assume_unique=True)
    return best_inds, val_inds


@torch.inference_mode()
def test(loader, model, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        acc = output.argmax(dim=1).eq(labels).sum().item()
        correct += acc
    return correct


# write a function validate that takes in a loader, model, criterion, and device and returns the mean loss and mean accuracy of the model on the validation set
@torch.inference_mode()
def validate(loader, model, criterion, device):
    model.eval()
    loss = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss += criterion(output, labels).item()
        correct += output.argmax(dim=1).eq(labels).sum().item()
    return loss / len(loader.dataset), correct / len(loader.dataset)


# def train_epoch(loader, model, criterion, optimizer, scheduler, device):
#     model.train()
#     optimizer.zero_grad(set_to_none=True)
#     for (images, labels) in loader:
#         images, labels = images.to(device), labels.to(device)
#         output = model(images)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#         acc = output.argmax(dim=1).eq(labels).float().mean().item()
#     return loss, acc


# write a function train_one_epoch which takes in a train_loader, val_loader, model, criterion, optimizer, device and trains the model for single epoch
def train_epoch(train_loader, model, criterion, optimizer, device):
    model.train()
    train_loss = 0
    train_correct = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        train_loss += loss.item()
        train_correct += output.argmax(dim=1).eq(labels).sum().item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    train_loss /= len(train_loader.dataset)
    train_correct /= len(train_loader.dataset)
    return train_loss, train_correct


def train_loop(p, best_inds: torch.Tensor, data, test_data) -> None:
    """Training Loop

    Args:
        p (EasyDict): Hyperparameters
        best_inds (torch.Tensor): indices for images in the coreset
        data (torch.utils.data.Dataset): Train Dataset
        test_data (torch.utils.data.Dataset): Test Dataset
    """
    train_inds, val_inds = get_train_val_inds(len(data), best_inds)

    # train_loader = DataLoader(Subset(data, train_inds), train_inds.shape[0], shuffle=True)
    train_loader = DataLoader(Subset(data, train_inds), p.batch_size, shuffle=True)
    p.len_loader = len(train_loader)
    val_loader = None
    # val_inds = None # TODO: remove this line
    if val_inds is not None:
        val_loader = DataLoader(Subset(data, val_inds), p.val_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, p.val_batch_size)

    # model
    model = get_model(p, device)
    logger.info(
        "Model Summary\n"
        + str(summary(model, data[1][0].shape, verbose=0, device=device))
    )

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = get_optimizer(p, model)
    scheduler = None
    if p.scheduler:
        scheduler = get_scheduler(p, optimizer)

    if p.early_stopping_patience == -1:
        early_stopping = None
    else:
        early_stopping = EarlyStopping(
            p.early_stopping_patience,
            p.early_stopping_delta,
            p.early_stopping_min_epochs,
        )
    losses, accs, val_losses, val_accs = [], [], [], []
    test_accs = []
    val_loss, val_acc = 0, 0
    lrs = []
    for epoch in trange(p.epochs, position=0, leave=True):
        loss, acc = train_epoch(train_loader, model, criterion, optimizer, device)
        losses.append(loss)
        accs.append(acc)

        if scheduler is not None:
            # make conditional if else for scheduler.step() if it requires a parameter

            scheduler.step()
            # scheduler.step(val_loss)
            lrs.append(optimizer.param_groups[0]["lr"])
        # logger.info(f"Epoch[{epoch+1:4}] Val_Loss: {val_loss:.3f}\tVal_Acc: {val_acc:.3f}")
        gc.collect()
        torch.cuda.empty_cache()
        if epoch % 10 == 0:
            if val_loader is not None:
                val_loss, val_acc = validate(val_loader, model, criterion, device)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                if early_stopping is not None:
                    early_stopping(-val_loss)

            correct = test(test_loader, model, device)
            test_accs.append(correct / len(test_data))
            logger.info(
                f"Epoch[{epoch+1:4}] Loss: {loss:.2f}\tAccuracy: {acc*100 :.3f}\tVal_Loss: {val_loss:.3f}\tVal_Acc: {val_acc:.3f}"
            )
            logger.info(
                f"Epoch[{epoch+1:4}] Test Accuracy: {(correct / len(test_data))*100 :.3f}"
            )
        if early_stopping is not None:
            if early_stopping.early_stop:
                logger.info(f"Trained for {epoch+1} Epochs.")
                break

    suffix = str("_augment" if p.augment else "") + str(
        "_clsbalanced" if p.class_balanced else "_perclass" if p.per_class else ""
    )
    prefix = "greedy"
    if p.random:
        prefix = "random" + str(train_loop.counter)
        train_loop.counter += 1
    # write a function to plot learning curves for loss, acc, val_loss, val_acc where val_loss and val_acc are after 5 epochs. loss and val_loss to be plot together and acc and val_acc to be plot together

    plot_loss_acc(
        losses,
        accs,
        val_losses,
        val_accs,
        test_accs,
        p.output_dir
        / f"{'random/' if p.random else ''}LearningCurve_{prefix}_n{p.topn}{suffix}",
    )
    if len(lrs) > 0:
        # plot learning rate wrt epochs
        plot_lr(
            lrs,
            p.output_dir
            / f"{'random/' if p.random else ''}Learningrate_{p.scheduler}_{prefix}_n{p.topn}{suffix}",
        )

    # model.eval()
    _, train_acc = validate(train_loader, model, criterion, device)
    logger.info(("Accuracy on Train Set", train_acc))
    correct = test(test_loader, model, device)
    logger.info((correct, "correctly labeled out of", len(test_data)))
    test_acc = correct / len(test_data) * 100
    logger.info(("Accuracy on Test Set:", test_acc))

    model_path = (
        p.output_dir
        / f"{'random/' if p.random else ''}Greedy_Model_{p.topn}n_Epochs_{p.epochs}_Test_Acc_{int(test_acc)}{suffix}.pth"
    )
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model at {str(model_path)}")
    logger.info("Training Complete")
    return train_acc * 100, test_acc


def main(p):
    logger.info("Hyperparameters\n" + pformat(vars(p)))

    global device
    if torch.cuda.is_available:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.warning("Using CPU to run the program.")

    seed_everything(p.seed)

    # dataset
    data, test_data = get_dataset(p)
    logger.info(f"Dataset\n{str(data)}")
    logger.info(f"Test Dataset\n{str(test_data)}")
    train_labels = np.array(data.targets)

    if p.test_model is not None:
        model = AlexNet(output_dim=p.num_classes, dropout=True).to(device)
        logger.info(f"Loading model from {p.test_model}")
        model.load_state_dict(torch.load(p.test_model, map_location=device))
        model.eval()
        test_loader = DataLoader(test_data, p.batch_size)
        correct = test(test_loader, model, device)
        logger.info((correct, "correctly labeled out of", len(test_data)))
        test_acc = correct / len(test_data) * 100
        logger.info(("Accuracy on Test Set:", test_acc))
        return

    if p.use_saved_best_inds is not None:
        logger.info(f"Loading best_indices from {str(p.use_saved_best_inds)}")
        best_inds = np.load(p.use_saved_best_inds)
        assert (
            best_inds.shape[0] == p.topn
        ), f"Given best indices shape {best_inds.shape[0]} and no. of best samples {p.topn} does not match."

    elif p.per_class:
        all_sim_path = (
            args.grad_path
            / f"all_similarities_perclass{'_withtrain' if p.with_train else ''}.npy"
        )
        all_ind_path = (
            args.grad_path
            / f"all_imginds_perclass{'_withtrain' if p.with_train else ''}.npy"
        )
        if p.dataset.lower() in ["cifar10", "cifar100"]:
            all_similarities = np.load(all_sim_path).squeeze()
            all_imginds = np.load(all_ind_path).astype(int)
        else:
            all_similarities = np.load(all_sim_path, allow_pickle=True)
            all_imginds = np.load(all_ind_path, allow_pickle=True)
        all_similarities = all_similarities.swapaxes(0, 1)
        all_imginds = all_imginds.swapaxes(0, 1)
        logger.info(
            f"all_similarities_perclass.shape: {all_similarities.shape}, all_imginds_perclass.shape: {all_imginds.shape}"
        )
        best_inds = []
        for i in range(all_similarities.shape[0]):
            # logger.debug(np.unique(train_labels[all_imginds[i]], return_counts=True))
            inds = get_best_inds(
                p.topn // p.num_classes, all_similarities[i], all_imginds[i]
            )
            best_inds.append(inds)
            # logger.debug(np.unique(train_labels[inds], return_counts=True))
        best_inds = np.concatenate(best_inds)
        logger.debug(f"best inds shape {best_inds.shape}")
        np.save(p.output_dir / f"best_inds_{p.topn}_perclass.npy", best_inds)
        plot_distribution(
            p.topn,
            train_labels[best_inds],
            data.classes,
            p.output_dir / f"freq_{p.topn}_perclass",
        )

    elif p.random:
        rand_iter = 10
        logger.info(f"Training on randomly chosen Coreset for {rand_iter} iterations.")
        train_loop.counter = 0
        rand_train_acc, rand_test_acc = [], []
        for i in range(rand_iter):
            np.random.seed(p.seed + i)
            if p.class_balanced:
                best_inds = np.concatenate(
                    [
                        np.random.choice(
                            np.argwhere(train_labels == c).squeeze(),
                            p.topn // p.num_classes,
                        )
                        for c in data.class_to_idx.values()
                    ]
                )
                plot_distribution(
                    p.topn,
                    train_labels[best_inds],
                    data.classes,
                    p.output_dir / f"freq_{p.topn}_random_clsbalanced",
                )
            else:
                best_inds = np.random.randint(0, len(data), p.topn)
                plot_distribution(
                    p.topn,
                    train_labels[best_inds],
                    data.classes,
                    p.output_dir / f"freq_{p.topn}_random",
                )
            best_inds = torch.from_numpy(best_inds)
            train_acc, test_acc = train_loop(p, best_inds, data, test_data)
            rand_train_acc.append(train_acc)
            rand_test_acc.append(test_acc)
        logger.info(
            f"Mean Train Accuracy on Random {p.topn} Train Samples is {np.mean(rand_train_acc):.3f}±{np.std(rand_train_acc):.2f}%"
        )
        logger.info(
            f"Mean Test Accuracy on Random {p.topn} Train Samples is {np.mean(rand_test_acc):.3f}±{np.std(rand_test_acc):.2f}%"
        )

    else:
        all_sim_path = (
            args.grad_path
            / f"all_similarities{'_withtrain' if p.with_train else ''}.npy"
        )
        all_ind_path = (
            args.grad_path / f"all_imginds{'_withtrain' if p.with_train else ''}.npy"
        )
        logger.info(
            f"Loading similarities from {all_sim_path} and imginds from {all_ind_path}"
        )
        all_similarities = np.load(all_sim_path).squeeze()
        all_imginds = np.load(all_ind_path).astype(int)
        logger.info(
            f"all_similarities.shape: {all_similarities.shape}, all_imginds.shape: {all_imginds.shape}"
        )

        if p.class_balanced:
            best_inds = get_cls_balanced_best_inds(
                p.topn, p.num_classes, train_labels, all_similarities, all_imginds
            )
            plot_distribution(
                p.topn,
                train_labels[best_inds],
                data.classes,
                p.output_dir / f"freq_{p.topn}_clsbalanced",
            )
        else:
            best_inds = get_best_inds(p.topn, all_similarities, all_imginds)
            plot_distribution(
                p.topn,
                train_labels[best_inds],
                data.classes,
                p.output_dir / f"freq_{p.topn}",
            )
        np.save(p.output_dir / f"best_inds_{p.topn}.npy", best_inds)

    if not (not p.train or p.random):
        best_inds = torch.from_numpy(best_inds)
        train_loop(p, best_inds, data, test_data)

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Getting gradient similarity for each sample."
    # )
    # parser.add_argument("--config", required=True, help="Location of config file")
    # parser.add_argument("--seed", default=0, type=int, help="Seed")
    # parser.add_argument(
    #     "--dataset", default="cifar100", required=True, help="Dataset Location"
    # )
    # parser.add_argument("--dataset_dir", default="./data", help="Dataset directory")
    # parser.add_argument("--topn", default=1000, type=int, help="Size of Coreset")
    # parser.add_argument(
    #     "--class_balanced",
    #     action="store_true",
    #     help="Specify to use class balanced distribution for training",
    # )
    # parser.add_argument(
    #     "--per_class",
    #     action="store_true",
    #     help="Specify whether to find Mean Gradients classwise",
    # )
    # parser.add_argument(
    #     "-bi",
    #     "--use_saved_best_inds",
    #     default=None,
    #     help="Specify path of already retreived best indices",
    # )
    # parser.add_argument(
    #     "--test_model", default=None, help="Specify path of model which is to be tested"
    # )
    # parser.add_argument(
    #     "--random",
    #     action="store_true",
    #     help="Specify if randomly chosen coreset to be used for training",
    # )
    # parser.add_argument(
    #     "--dont_train",
    #     action="store_true",
    #     help="Specify is model need not to be trained",
    # )
    # parser.add_argument("-bs", "--batch_size", default=1000, type=int, help="BatchSize")
    # parser.add_argument(
    #     "-v",
    #     "--val_percent",
    #     default=0.1,
    #     type=float,
    #     help="Percentage[0-1] split of Validation set. (Default: 0.1)",
    # )
    # parser.add_argument(
    #     "--augment",
    #     action="store_true",
    #     help="Specify to use augmentation during training",
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
    #     "--resume", default=None, help="path to checkpoint from where to resume"
    # )
    # parser.add_argument("--wandb", default=False, type=bool, help="Log using wandb")
    # parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs)

    parser = get_parser()
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir = Path(args.dataset.lower()) / args.output_dir
    else:
        args.output_dir = Path(args.dataset.lower())
    if args.pretrained:
        args.output_dir = args.output_dir / "pretrained"
    if args.temp:
        args.output_dir = args.output_dir / f"temp"
    args.grad_path = args.output_dir
    if args.with_train:
        args.output_dir = args.output_dir / f"with_train"
    args.output_dir = args.output_dir / f"n{args.topn}"
    args.logdir = args.output_dir / "logs"
    args.logdir.mkdir(parents=True, exist_ok=True)

    if args.test_model is not None and not Path(args.test_model).is_file():
        raise ValueError("Provided path to model does not exists.")

    args.use_saved_best_inds = None
    if (
        args.use_saved_best_inds is not None
        and not Path(args.use_saved_best_inds).is_file()
    ):
        raise ValueError("Best indices file does not exist.")

    if args.random:
        (args.output_dir / "random").mkdir(exist_ok=True)

    global logger
    logger = get_logger(args, "train")

    # log the command used to run the program and mention the Command:
    logger.info(f"Command: {' '.join(sys.argv)}")

    args.val_percent = 0.0
    try:
        main(args)
    except Exception:
        logger.exception("An Error Occurred")
