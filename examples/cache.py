import os
import time
import datetime

import toml
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

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
        else:
            self.counter += 1
        
        if self.counter > self.patience:
            self.stop_early == True
    
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
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)

            for key, value in targets.items():
                targets[key] = targets[key].to(device)

            output = model(data)
            loss = model.loss_value(data, targets, output, reduce='sum')
            history.loss_meter.add_val_loss(loss.item())
            val_loss += loss

    history.patience_meter.check_loss(val_loss)
    val_loss /= len(val_loader.dataset)
    print(f'\nValidation set: Average loss: {val_loss:.4f}\n')


def main():
    parser = argparse.ArgumentParser(description='MTCNN P3B3')
    parser.add_argument('--datapath', '-p', type=str, default='/home/ygx/data', help='Path to data.')
    parser.add_argument('--batchsize', '-bs', type=int, default=20, help='Batch size.')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=10, help='interval to log.')
    parser.add_argument('--savepath', type=str, default='/home/ygx/src/bayescache/examples')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:3" if use_cuda else "cpu")

    traindata = P3B3(args.datapath, partition='train', download=True)
    valdata = P3B3(args.datapath, partition='test', download=True)

    train_loader = DataLoader(traindata, batch_size=args.batchsize)
    val_loader = DataLoader(valdata, batch_size=args.batchsize)

    model = mtcnn.new()
    model = model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=7.0e-4, eps=1e-3)
    history = OptimizationHistory(savepath=args.savepath, filename='history.toml')

    for opt in range(1, 3):
        for epoch in range(1, args.epochs + 1):
            history.epoch_meter.increment()
            train(args, model, device, train_loader, optimizer, epoch, history)
            test(args, model, device, val_loader, history)
            if history.patience_meter.stop_early:
                print(f'Patience exceeded, stopping early!')
                break
           
        history.time_meter.stop_timer()
        history.record_history()
        history.reset_meters()
        history.save()

    print(f'\n--- History ---')
    print(f'Runtime: {history.runtime}\n')
    print(f'Epochs: {history.num_epochs}\n')
    print(f'Stop Epoch: {history.stop_epoch}')
    print(f'Train Loss: {len(history.train_loss)}, {np.array(history.train_loss[-1]).shape}\n')
    print(f'Val Loss: {len(history.val_loss)}, {np.array(history.val_loss[-1]).shape}')


if __name__=='__main__':
    main()