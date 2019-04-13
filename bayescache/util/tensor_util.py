import torch


def one_hot_encoding(input_tensor, num_labels):
    """ One-hot encode labels from input """
    xview = input_tensor.view(-1, 1).to(torch.long)

    onehot = torch.zeros(xview.size(0), num_labels, device=input_tensor.device, dtype=torch.float)
    onehot.scatter_(1, xview, 1)
    return onehot.view(list(input_tensor.shape) + [-1])

