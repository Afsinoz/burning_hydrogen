import numpy as np

# Normalize the values in dataset.npy and replace nan values
# for use in neural networks

# Load the dataset
dataset = np.load('dataset.npy')
dataset_normalized = dataset.copy()

# Normalize each channel of dataset_normalized separately,
# ignoring nan
for i in range(8):
    dataset_normalized[:,:,i,:,:] = \
    (dataset_normalized[:,:,i,:,:] - np.nanmean(dataset_normalized[:,:,i,:,:])) / \
    np.nanstd(dataset_normalized[:,:,i,:,:])

# Replace nan values with zero, convert array to float32
# for use with PyTorch
dataset_normalized = np.nan_to_num(dataset_normalized, nan=0.)
dataset_normalized = dataset_normalized.astype('float32')
np.save('dataset_normalized.npy', dataset_normalized)
