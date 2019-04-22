from bayescache.data.p3b3 import P3B3
from bayescache.api import TrainingData


def create(model_config, batch_size, num_workers=0, augmentations=None):
    """ Create an Anon dataset."""
    path = model_config.data_dir('p3b3')

    train_dataset = P3B3(path, partition='train', download=True)
    test_dataset = P3B3(path, partition='test', download=True)

    return TrainingData(
        train_dataset,
        test_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        augmentations=augmentations
    )