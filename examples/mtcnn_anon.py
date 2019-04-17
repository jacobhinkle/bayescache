import argparse

import torch.optim as optim
import torch.nn.functional as F

import bayescache.api as api
from bayescache.data import P3B3
from bayescache.models import mtcnn
from bayescache.api.source import TrainingData
from bayescache.callbacks.time_tracker import TimeTracker


def main():
    parser = argparse.ArgumentParser(description='MTCNN P3B3')
    parser.add_argument('--datapath', '-p', type=str,
                        default='/home/ygx/data', help='Path to data.')
    args = parser.parse_args()

    hparams = mtcnn.Hyperparameters()
    # Update hyperparameters for the Synthetic data.
    hparams.vocab_size = 4014
    hparams.max_sent_len = 1500

    model = mtcnn.new(hparams)
    learner = api.Learner(device='cpu', model=model)
    optimizer = optim.RMSprop(model.parameters(), lr=7.0e-4, eps=1e-3)

    train = P3B3(root=args.datapath, partition='train', download=True)
    test = P3B3(root=args.datapath, partition='test', download=True)
    source = TrainingData(train_source=train, val_source=test, num_workers=2, batch_size=4)

    metrics = learner.metrics()
    callbacks = [TimeTracker()]

    training_info = api.TrainingInfo(
        start_epoch_idx=0,
        run_name='test',
        metrics=metrics,
        callbacks=callbacks
    )

    training_info.on_train_begin()

    for global_epoch_idx in range(training_info.start_epoch_idx + 1, 1 + 1):
        epoch_info = api.EpochInfo(
            training_info=training_info,
            global_epoch_idx=global_epoch_idx,
            batches_per_epoch=source.train_iterations_per_epoch(),
            optimizer=optimizer
        )

        # Execute learning
        learner.train_epoch(epoch_info, source)


if __name__=='__main__':
    main()