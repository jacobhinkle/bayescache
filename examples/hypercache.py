import os
import time
import datetime

import toml
import json
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from skopt import gp_minimize
from skopt.space.space import Space
from hyperspace.rover.checkpoints import JsonCheckpointSaver

from bayescache.data import P3B3
from bayescache.models import mtcnn


class TimeMeter:
    """Measure time"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.elapsed_time = 0.0
        self.time = time.time()

    def stop_timer(self):
        self.elapsed_time = time.time() - self.time

    def get_timings(self):
        return self.elapsed_time


class EpochMeter:
    """Count epochs"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def get_counts(self):
        return self.n


class LossMeter:
    """Record training and validation loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.train_loss = []
        self.val_loss = []

    def add_train_loss(self, loss):
        self.train_loss.append(loss)

    def add_val_loss(self, loss):
        self.val_loss.append(loss)

    def get_train_loss(self):
        return self.train_loss

    def get_val_loss(self):
        return self.val_loss


class PatienceMeter:
    """Patience for validation improvement"""
    def __init__(self, patience=5):
        self.patience = patience
        self.reset()

    def reset(self):
        self.counter = 0
        self.epoch = 0
        self.minimal_loss = 9e10
        self.stop_early = False

    def check_loss(self, loss):
        self.epoch += 1

        if loss < self.minimal_loss:
            self.minimal_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter > self.patience:
            self.stop_early = True

    def get_stop_epoch(self):
        return self.epoch


class OptimizationHistory:
    """Records of the Bayesian optimization"""
    def __init__(self, savepath=None, filename=None):
        self.time_meter = TimeMeter()
        self.epoch_meter = EpochMeter()
        self.loss_meter = LossMeter()
        self.patience_meter = PatienceMeter()
        self.savepath = savepath
        self.filename = filename
        self.reset()

    def reset(self):
        self.runtime = []
        self.num_epochs = []
        self.train_loss = []
        self.val_loss = []
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
        self.val_loss.append(self.loss_meter.get_val_loss())
        self.stop_epoch.append(self.patience_meter.get_stop_epoch())

    def save(self):
        if self.savepath == None or self.filename == None:
            raise ValueError("You must specify a savepath and filename to save results.")

        now = datetime.datetime.now()

        history = {
            'Title': self.filename,
            'Date': now.strftime("%Y-%m-%d"),
            'NumEpochs': self.num_epochs,
            'StopEpoch': self.stop_epoch,
            'Runtime': self.runtime,
            'TrainLoss': self.train_loss,
            'ValidationLoss': self.val_loss,
        }

        savefile = os.path.join(self.savepath, self.filename)
        with open(savefile, 'w') as outfile:
            toml.dump(history, outfile)


def train(args, model, device, train_loader, optimizer, epoch, history):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)

        for key, value in targets.items():
            targets[key] = targets[key].to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_value(data, targets, output, reduce='sum')
        history.loss_meter.add_train_loss(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, val_loader, history):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)

            for key, value in targets.items():
                targets[key] = targets[key].to(device)

            output = model(data)
            loss = model.loss_value(data, targets, output, reduce='sum')
            history.loss_meter.add_val_loss(loss.item())
            valid_loss += loss

    valid_loss /= len(val_loader.dataset)
    valid_loss = round(valid_loss.item(), 5)
    history.patience_meter.check_loss(valid_loss)

    print(f'\nValidation set: Average loss: {valid_loss:.4f}\n')
    return valid_loss


def objective(hparams, args, device, train_loader, val_loader, history):
    hyperparameters = mtcnn.Hyperparameters()
    hyperparameters.kernel1 = hparams[0]
    hyperparameters.kernel2 = hparams[1]

    try:
        modelstate = torch.load(args.modelstate)
        model = mtcnn.new(hyperparameters=hyperparameters, savefile=modelstate)
    except:
        print(f'No previous model state to load. Starting fresh!')
        model = mtcnn.new(hyperparameters=hyperparameters)

    model = model.to(device)
    optimizer = optim.Adadelta(model.parameters())

    for epoch in range(1, args.epochs + 1):
        history.epoch_meter.increment()
        train(args, model, device, train_loader, optimizer, epoch, history)
        valid_loss = test(args, model, device, val_loader, history)
        if history.patience_meter.stop_early:
            print(f'Patience exceeded, stopping early!')
            break

    history.time_meter.stop_timer()
    history.record_history()
    history.reset_meters()
    history.save()

    cache_model = not args.no_cache

    if cache_model:
        print(f'Caching model!')
        torch.save(model.state_dict(), args.modelstate)

    return valid_loss


def main():
    parser = argparse.ArgumentParser(description='MTCNN P3B3')
    parser.add_argument('--datapath', '-p', type=str, default='/home/ygx/data', help='Path to data.')
    parser.add_argument('--batchsize', '-bs', type=int, default=20, help='Batch size.')
    parser.add_argument('--epochs', '-e', type=int, default=25, help='Number of epochs.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=10, help='interval to log.')
    parser.add_argument('--savepath', type=str, default='/home/ygx/src/bayescache/examples')
    parser.add_argument('--modelstate', type=str, default='/home/ygx/src/bayescache/examples/hyperstate')
    parser.add_argument('--bayescheckpoint', type=str, default='/home/ygx/src/bayescache/examples/checkpoint.pkl')
    parser.add_argument('--no_cache', action='store_true', default=False, help='Disables model cache')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:3" if use_cuda else "cpu")

    traindata = P3B3(args.datapath, partition='train', download=True)
    valdata = P3B3(args.datapath, partition='test', download=True)

    train_loader = DataLoader(traindata, batch_size=args.batchsize)
    val_loader = DataLoader(valdata, batch_size=args.batchsize)

    history = OptimizationHistory(savepath=args.savepath, filename='history_cache.toml')
    checkpoint_file = os.path.join(args.savepath, f'bayes_cache_checkpoint')
    checkpoint_saver = JsonCheckpointSaver(args.savepath, f'bayes_cache_checkpoint')

    search_bounds = [
        (2, 6),  # kernel1
        (2, 6)   # kernel2
    ]

    try:
        with open(checkpoint_file) as infile:  
            res = json.load(infile)
        x0 = res['x_iters']
        y0 = res['func_vals']
    except FileNotFoundError:
        print(f'No previous save point for Bayesian optimization. Starting fresh!')
        # Need to randomly sample the bounds to prime the optimization.
        space = Space(search_bounds)
        x0 = space.rvs(1)
        y0 = None

    gp_minimize(
        lambda x: objective(
            x,
            args,
            device,
            train_loader,
            val_loader,
            history
        ),
        search_bounds,
        x0=x0,
        y0=y0,  
        acq_func="LCB",
        n_calls=20,
        n_random_starts=0,
        callback=[checkpoint_saver],
        random_state=777
    )

    print(f'\n--- History ---')
    print(f'Runtime: {history.runtime}\n')
    print(f'Epochs: {history.num_epochs}\n')
    print(f'Stop Epoch: {history.stop_epoch}')
    print(f'Train Loss: {len(history.train_loss)}, {np.array(history.train_loss[-1]).shape}\n')
    print(f'Val Loss: {len(history.val_loss)}, {np.array(history.val_loss[-1]).shape}')


if __name__=='__main__':
    main()
