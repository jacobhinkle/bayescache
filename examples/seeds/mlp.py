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
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from mpi4py import MPI

from bayescache.models import mlp
from bayescache.metrics import accuracy_topk
from bayescache.util.random import SeedControl
from bayescache.meters import OptimizationHistory


def train(args, model, device, train_loader, optimizer, epoch, history):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_value(data, target, output)
        history.loss_meter.add_train_loss(loss.item())
        acc1, acc2 = accuracy_topk(output, target, topk=(1, 2))
        history.top1_train.update(acc1[0], data.size(0))
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Acc@1: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                acc1[0]))


def test(args, model, device, val_loader, history):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.view(-1, 784)
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            acc1, acc2 = accuracy_topk(output, target, topk=(1, 2))
            history.top1_valid.update(acc1[0], data.size(0))
            loss = model.loss_value(data, target, output)
            history.loss_meter.add_valid_loss(loss.item())
            valid_loss += loss

    valid_loss /= len(val_loader.dataset)
    valid_loss = round(valid_loss.item(), 5)

    print(f'\nValidation set: Average loss: {valid_loss:.4f}\n, Acc@1: {history.top1_valid.avg}')
    return valid_loss


def main():
    parser = argparse.ArgumentParser(description='MTCNN P3B3')
    parser.add_argument('--datapath', '-p', type=str, default='/Users/yngtodd/data', help='Path to data.')
    parser.add_argument('--batchsize', '-bs', type=int, default=20, help='Batch size.')
    parser.add_argument('--epochs', '-e', type=int, default=25, help='Number of epochs.')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=10, help='interval to log.')
    parser.add_argument('--savepath', type=str, default='/Users/yngtodd/src/ornl/bayescache/output/seeds/shuffle_mlp')
    parser.add_argument('--no_cache', action='store_true', default=False, help='Disables model cache')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the data')
    parser.add_argument('--num_workers', type=int, default=0, help='num threads for data loaders.')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    seedcontrol = SeedControl()
    seedcontrol.fix_all_seeds(rank)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    traindata = FashionMNIST(args.datapath, train=True, download=True, transform=transforms.ToTensor())
    valdata = FashionMNIST(args.datapath, train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(
        traindata, 
        batch_size=args.batchsize, 
        shuffle=args.shuffle, 
        num_workers=args.num_workers
    )
   
    val_loader = DataLoader(
        valdata, 
        batch_size=args.batchsize, 
        shuffle=args.shuffle, 
        num_workers=args.num_workers
    )

    dataloader_info = {
        'shuffle': args.shuffle,
        'num_workers': args.num_workers
    }

    history = OptimizationHistory(
        savepath=args.savepath, 
        experiment_name='mlp_fashionmnist', 
        device = device,
        dataloader_info = dataloader_info,
        seeds = seedcontrol.get_seeds(),
        rank=rank
    )

    model = mlp.new(input_size=784)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, args.epochs + 1):
        history.epoch_meter.increment()
        train(args, model, device, train_loader, optimizer, epoch, history)
        valid_loss = test(args, model, device, val_loader, history)

    history.time_meter.stop_timer()
    history.record_history()
    history.reset_meters()
    history.save()

    print(f'\n--- History ---')
    print(f'Runtime: {history.runtime}\n')
    print(f'Epochs: {history.num_epochs}\n')
    print(f'Stop Epoch: {history.stop_epoch}')
    print(f'Train Loss: {len(history.train_loss)}, {np.array(history.train_loss[-1]).shape}\n')
    print(f'Val Loss: {len(history.valid_loss)}, {np.array(history.valid_loss[-1]).shape}')


if __name__=='__main__':
    main()
