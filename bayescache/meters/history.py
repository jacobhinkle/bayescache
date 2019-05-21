import os
import toml
import time
import datetime

import pandas as pd

from bayescache.meters import (
    EpochMeter, LossMeter, PatienceMeter, TimeMeter
)


class OptimizationHistory:
    """Records of the optimization"""
    def __init__(self, savepath=None, experiment_name=None, rank=0):
        self.time_meter = TimeMeter()
        self.epoch_meter = EpochMeter()
        self.loss_meter = LossMeter()
        self.patience_meter = PatienceMeter()
        self.savepath = savepath
        self.experiment_name = experiment_name
        self.rank = rank
        self.reset()

    def reset(self):
        self.runtime = []
        self.num_epochs = []
        self.train_loss = []
        self.valid_loss = []
        self.stop_epoch = []

    def reset_meters(self):
        self.time_meter.reset()
        self.epoch_meter.reset()
        self.loss_meter.reset()
        self.patience_meter.reset()

    def record_history(self):
        self.runtime.append(self.time_meter.get_timings())
        self.num_epochs.append(self.epoch_meter.get_counts())
        self.train_loss.append(self.loss_meter.get_train_loss())
        self.valid_loss.append(self.loss_meter.get_valid_loss())
        self.stop_epoch.append(self.patience_meter.get_stop_epoch())

    def save_metadata(self):
        if self.savepath == None or self.experiment_name == None:
            raise ValueError("You must specify a savepath and experiment name to save results.")

        now = datetime.datetime.now()

        metadata = {
            'Title': self.experiment_name,
            'Date': now.strftime("%Y-%m-%d"),
            'NumEpochs': self.num_epochs,
            'StopEpoch': self.stop_epoch,
            'Runtime': self.runtime,
        }

        metafile = os.path.join(self.savepath, 'metadata.toml')
        with open(metafile, 'w') as outfile:
            toml.dump(metadata, outfile)

    def save(self):
        self.save_metadata()

        # Create separate directories to keep train/valid loss for all ranks
        savepath_train = os.path.join(self.savepath, 'trainloss')
        savepath_valid = os.path.join(self.savepath, 'validloss')
        os.makedirs(savepath_train, exist_ok=True)
        os.makedirs(savepath_valid, exist_ok=True)

        trainsave = os.path.join(savepath_train, f'trainloss{self.rank}.csv')
        validsave = os.path.join(savepath_valid, f'validloss{self.rank}.csv')

        trainloss_df = pd.DataFrame({'loss': self.train_loss[0]})
        validloss_df = pd.DataFrame({'loss': self.valid_loss[0]})

        trainloss_df.to_csv(trainsave, index=False)
        validloss_df.to_csv(validsave, index=False)
