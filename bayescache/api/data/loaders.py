import numpy as np
from math import floor

from torch.utils import data


class DataSplitter(data.Dataset):
    """
    Subset dataset by index.

    Helper class to be used with `train_valid_splitter`.

    Parameters
    ----------
    data : torch.utils.data.Dataset instance

    length : int
        Number of samples in subset.

    mapping : list
        Indices of the original data to be used in subset.
    """
    def __init__(self, data, length, mapping):
        self.data = data
        self.length = length
        self.mapping = mapping

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, index):
        return self.data[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(data, valpercent=.20, random_seed=None):
    """
    Split dataset into train and validation sets.

    Parameters
    ----------
    data : torch.utils.data.DataSet instance
        Dataset to be split into training and validation sets.

    valpercent : float
        Percentage of the validation set to be withheld for validation.
        Note: will take the floor of that percentage, we can't index by floats.

    random_seed : int
        Random seed for shuffling.

    Returns
    -------
    train : torch.utils.data.Dataset instance
        Training set.

    valid : torch.utils.data.Dataset instance
        Validation set.
    """
    if random_seed!=None:
        np.random.seed(random_seed)

    datalen = len(data)
    valid_size = floor(datalen * valpercent)
    train_size = datalen - valid_size

    indices = list(range(datalen))
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]

    train = DataSplitter(data, train_size, train_mapping)
    valid = DataSplitter(data, valid_size, valid_mapping)

#    train.__repr__ = update_repr(train, 'train', len(train))
#    valid.__repr__ = update_repr(valid, 'valid', len(valid))

    return train, valid


def update_repr(data, partition, n_samples):
    fmt_str = 'Dataset ' + data.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(n_samples)
    fmt_str += '    Split: {}\n'.format(partition)
    fmt_str += '    Root Location: {}\n'.format(data.data.root)
    return fmt_str
