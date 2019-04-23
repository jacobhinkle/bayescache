import argparse

import torch.optim as optim
from torch.utils.data import DataLoader

from bayescache.data import P3B3
from bayescache.models import mtcnn


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)

        for key, value in targets.item():
            targets[key] = target[key].to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = model.loss_value(data, targets, output, reduce='sum')
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, val_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data = data.to(device)

            for key, value in targets.item():
                targets[key] = targets[key].to(device)

            output = model(data)
            loss = model.loss_value(data, targets, output, reduce='sum')
            val_loss += loss

    val_loss /= len(val_loader.dataset)
    print(f'\nValidation set: Average loss: {val_loss:.4f}\n')


def main():
    parser = argparse.ArgumentParser(description='MTCNN P3B3')
    parser.add_argument('--datapath', '-p', type=str, default='/home/ygx/data', help='Path to data.')
    parser.add_argument('--batchsize', '-bs', type=int, default=28, help='Batch size.')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs.')
    args = parser.parse_args()

    train = P3B3(args.datapath, partition='train', download=True)
    val = P3B3(args.datapath, partition='test', download=True)

    train_loader = DataLoader(train, batch_size=args.batchsize)
    val_loader = DataLoader(val, batch_size=args.batchsize)

    model = mtcnn.new()
    optimizer = optim.RMSprop(model.parameters(), lr=7.0e-4, eps=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, val_loader)
