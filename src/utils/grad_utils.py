import numpy as np

all_similarities = np.load('/home/ivlabs/Documents/ABD/Coreset/svhn/all_similarities_perclass_withtrain.npy', allow_pickle=True).swapaxes(0, 1)
all_imgindices = np.load('/home/ivlabs/Documents/ABD/Coreset/svhn/all_imginds_perclass_withtrain.npy', allow_pickle=True).swapaxes(0, 1)

print(all_similarities.shape, all_imgindices.shape)
print(all_similarities[0][0].shape)
