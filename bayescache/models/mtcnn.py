import torch
import torch.nn as nn


class Hyperparameters:
    kernel1 = 3
    kernel2 = 4
    kernel3 = 5
    n_filters1 = 300
    n_filters2 = 300
    n_filters3 = 300
    vocab_size = 3000
    word_dim = 100


class Conv1d(nn.Module):
    """Module allows us to save the name of our layers for cache."""

    def __init__(self, out_channels, kernel_size):
        super(Conv1d, self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module(
            f"conv1d_{str(kernel_size)}_{str(out_channels)}",
            nn.Conv1d(1, out_channels, kernel_size)
        )

    def forward(self, x):
        return self.conv(x)


class MTCNN(nn.Module):

    def __init__(self, hparams):
        super(CNN, self).__init__()
        self.hp = hparams
        self.embedding = nn.Embedding(
            hparams.vocab_size + 2,
            hparams.word_dim,
            padding_idx=0
        )
        self.conv1 = Conv1d(hparams.n_filters1, hparams.kernel1)
        self.conv2 = Conv1d(hparams.n_filters2, hparams.kernel2)
        self.conv3 = Conv1d(hparams.n_filters3, hparams.kernel3)
        self.fc = nn.Linear(self._sum_filters(), 10)

    def _sum_filters(self):
        return self.hp.n_filters1 + self.hp.n_filters2 + self.hp.n_filters3

    def forward(self, x):
        x = self.embedding(x).view(-1, 1, self.word_dim * self.max_sent_len)

        conv_results = []
        conv_results.append(nn.ReLU(self.conv1(x)).view(-1, self.hp.n_filters1))
        conv_results.append(nn.ReLU(self.conv2(x)).view(-1, self.hp.n_filters2))
        conv_results.append(nn.ReLU(self.conv2(x)).view(-1, self.hp.n_filters3))
        x = torch.cat(conv_results, 1)

        x = self.fc(x)
        return x


def new(hyperparameters=None, savefile=None):
    """Create new MTCNN model."""
    if hyperparameters:
        hparams = hyperparameters
    else:
        hparams = Hyperparameters()

    model = CNN(hparams)

    if savefile:
        model.load_state_dict(savefile, strict=False)

    return model
