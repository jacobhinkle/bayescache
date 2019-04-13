from bayescache.data.anon import Anon
from bayescache.api import TrainingData


def create(model_config, batch_size, normalize=True, num_workers=0, augmentations=None):
    """ Create an Anon dataset."""
    path = model_config.data_dir('anon')

    train_dataset = Anon(path, partition='train', download=True)
    test_dataset = Anon(path, partition='test', download=True)

    return TrainingData(
        train_dataset,
        test_dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        augmentations=augmentations
    )
