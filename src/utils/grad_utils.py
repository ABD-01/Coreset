from tqdm import tqdm
import numpy as np
import os, sys

# import nets and datasets from the parent's parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import datasets


def get_best_inds(
    topn: int,
    all_similarities: np.ndarray,
    all_imginds: np.ndarray,
    train_labels: np.ndarray,
) -> np.ndarray:
    # from utils import get_train_dataset
    # train_labels = np.array(get_train_dataset(p).targets)
    # print((topn, all_similarities.shape, all_imginds.shape))
    print("train labels for all_imginds")
    print(np.unique(train_labels[all_imginds], return_counts=True))
    good_inds = []
    for (sims, inds) in tqdm(
        zip(all_similarities, all_imginds),
        total=len(all_similarities),
        desc="Getting best inds",
    ):
        inds = inds.astype(int)
        # print(sims.shape)
        ind = np.argpartition(-sims, topn)[:topn]
        # ind = np.argpartition(sims, topn)[:topn] # for least similar samples
        good_inds.append(inds[ind])
        print("train labels for ind")
        print(train_labels[inds[ind]])
        print(np.unique(train_labels[inds[ind]], return_counts=True))
    good_inds = np.concatenate(good_inds)
    print("train labels for good_inds")
    print(np.unique(train_labels[good_inds], return_counts=True))
    values, counts = np.unique(good_inds, return_counts=True)
    print((values, counts))
    # ref:https://stackoverflow.com/a/28736715/13730689
    best_inds = np.argpartition(-counts, kth=topn)[:topn]
    print("train labels for best_inds")
    print(np.unique(train_labels[best_inds], return_counts=True))
    print("train labels for good_inds[best_inds]")
    print(np.unique(train_labels[good_inds[best_inds]], return_counts=True))
    print(best_inds)
    return good_inds[best_inds]


all_similarities = np.load(
    "/home/ivlabs/Documents/ABD/Coreset/svhn/all_similarities.npy", allow_pickle=True
)  # .swapaxes(0, 1)
all_imgindices = np.load(
    "/home/ivlabs/Documents/ABD/Coreset/svhn/all_imginds.npy", allow_pickle=True
)  # .swapaxes(0, 1)

all_similarities = all_similarities.squeeze()
all_imgindices = all_imgindices.astype(int)

channel, im_size, num_classes, class_names, mean, std, data, _ = datasets.__dict__[
    "SVHN"
]("/home/ivlabs/Documents/ABD/Coreset/data")

train_labels = np.array(data.targets)
print(train_labels.shape)

best_inds = get_best_inds(500, all_similarities, all_imgindices, train_labels)
# print(best_inds)

best_labels = train_labels[best_inds]
unique_and_counts = np.unique(best_labels, return_counts=True)
# print(unique_and_counts)

# print(all_similarities.shape, all_imgindices.shape)
# print(all_similarities[0][0].shape)
