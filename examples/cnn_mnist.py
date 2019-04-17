import argparse

import torch.optim as optim
import torch.nn.functional as F

import bayescache.api as api
from bayescache.models import cnn
from bayescache.sources import mnist
from bayescache.api.source import TrainingData
from bayescache.callbacks.time_tracker import TimeTracker


def main():
    parser = argparse.ArgumentParser(description='CNN MNIST')
    parser.add_argument('--datapath', '-p', type=str,
                        default='/home/ygx/data', help='Path to data.')
    args = parser.parse_args()

    model = cnn.new()
    learner = api.Learner(device='cuda:0', model=model)
    source = mnist.new(args.datapath, batch_size=128)
    optimizer = optim.RMSprop(model.parameters(), lr=7.0e-4, eps=1e-3)

    metrics = learner.metrics()
    callbacks = []

    training_info = api.TrainingInfo(
        start_epoch_idx=0,
        run_name='test',
        metrics=metrics,
        callbacks=callbacks
    )

    training_info.on_train_begin()

    for global_epoch_idx in range(training_info.start_epoch_idx + 1, 1+10):
        epoch_info = api.EpochInfo(
            training_info=training_info,
            global_epoch_idx=global_epoch_idx,
            batches_per_epoch=source.train_iterations_per_epoch(),
            optimizer=optimizer
        )

        # Execute learning
        learner.run_epoch(epoch_info, source)


if __name__=='__main__':
    main()
