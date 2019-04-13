import torch.utils.data as data
from bayescache.api.data.loaders import train_valid_split


class Source:
    """ Source of data for supervised learning algorithms """
    def __init__(self):
        pass

    def train_loader(self):
        """ PyTorch loader of training data """
        raise NotImplementedError

    def val_loader(self):
        """ PyTorch loader of validation data """
        raise NotImplementedError

    def train_dataset(self):
        """ Return the training dataset """
        raise NotImplementedError

    def val_dataset(self):
        """ Return the validation dataset """
        raise NotImplementedError

    def train_iterations_per_epoch(self):
        """ Return number of iterations per epoch """
        raise NotImplementedError

    def val_iterations_per_epoch(self):
        """ Return number of iterations per epoch - validation """
        raise NotImplementedError


class TrainingData(Source):
    """ Most common source of data combining a basic datasource and sampler """
    def __init__(self, dataset, valpercent, num_workers, batch_size, augmentations=None, random_seed=None):
        import vel.api.data as vel_data

        super().__init__()

        self.dataset = dataset

        self.num_workers = num_workers
        self.batch_size = batch_size

        self.augmentations = augmentations

        # Derived values
        self.train, self.valid = train_valid_split(self.dataset, valpercent, random_seed=random_seed)

        self._train_loader = data.DataLoader(
            self.train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self._val_loader = data.DataLoader(
            self.valid, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    def train_loader(self):
        """ PyTorch loader of training data """
        return self._train_loader

    def val_loader(self):
        """ PyTorch loader of validation data """
        return self._val_loader

    def train_dataset(self):
        """ Return the training dataset """
        return self.train

    def val_dataset(self):
        """ Return the validation dataset """
        return self.valid

    def train_iterations_per_epoch(self):
        """ Return number of iterations per epoch """
        return len(self._train_loader)

    def val_iterations_per_epoch(self):
        """ Return number of iterations per epoch - validation """
        return len(self._val_loader)
