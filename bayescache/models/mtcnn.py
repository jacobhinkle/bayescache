import torch
import torch.nn as nn
import torch.nn.functional as F

from bayescache.metrics.loss_metric import Loss
from bayescache.metrics.accuracy import Accuracy
from bayescache.api import MultiTaskSupervisedModel, ModelFactory


class Hyperparameters:
    kernel1 = 3
    kernel2 = 4
    kernel3 = 5
    n_filters1 = 300
    n_filters2 = 300
    n_filters3 = 300
    vocab_size = 35095
    word_dim = 300
    max_sent_len=1500 


class Conv1d(nn.Module):
    """Wrapper allows us to save the name of our layers for cache."""

    def __init__(self, out_channels, kernel_size):
        super(Conv1d, self).__init__()

        self.conv = nn.Sequential()
        self.conv.add_module(
            f"conv1d_{str(kernel_size)}_{str(out_channels)}",
            nn.Conv1d(1, out_channels, kernel_size)
        )

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.adaptive_max_pool1d(x, output_size=1)
        return x


class Embedding(nn.Module):
    """Wrapper for embedding to save names in cache."""
    def __init__(self, vocab_size, word_dim):
        super(Embedding, self).__init__()

        self.emb = nn.Sequential()
        self.emb.add_module(
            f"embedding_{str(vocab_size)}_{str(word_dim)}",
            nn.Embedding(vocab_size, word_dim, padding_idx=0)
        )

    def forward(self, x):
        return self.embedding(x)


class MTCNN(MultiTaskSupervisedModel):

    def __init__(self, hparams, subsite_size=6, laterality_size=2,
                 histology_size=2, grade_size=3):
        super(MTCNN, self).__init__()
        self.hp = hparams
        self.embedding = Embedding(hparams.vocab_size, hparams.word_dim)
        self.conv1 = Conv1d(hparams.n_filters1, hparams.kernel1)
        self.conv2 = Conv1d(hparams.n_filters2, hparams.kernel2)
        self.conv3 = Conv1d(hparams.n_filters3, hparams.kernel3)
        # TODO: Check the names of these labels. -> must match data. 
        self.fc1 = nn.Linear(self._sum_filters(), subsite_size)
        self.fc2 = nn.Linear(self._sum_filters(), laterality_size)
        self.fc3 = nn.Linear(self._sum_filters(), histology_size)
        self.fc4 = nn.Linear(self._sum_filters(), grade_size)

    def _sum_filters(self):
        return self.hp.n_filters1 + self.hp.n_filters2 + self.hp.n_filters3

    def loss_value(self, x_data, y_true, y_pred, reduce=None):
        """ Calculate a value of loss function """
        y_pred = self(x_data)

        losses = {}
        for key, value in y_true.items():
            # TODO: test this bad boy.
            # y_true and y_pred must have the same keys.
            losses[key] = F.cross_entropy(F.softmax(y_pred[key]), y_true[key])

        if reduce:
            total = 0
            for _, value in losses.items():
                total += value
            
            if reduce == "mean":
                losses = total / len(losses)
            elif reduce == "sum":
                losses = total

        return losses

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]

    def forward(self, x):
        x = self.embedding.emb(x).view(-1, 1, self.hp.word_dim * self.hp.max_sent_len)

        conv_results = []
        conv_results.append(self.conv1(x).view(-1, self.hp.n_filters1))
        conv_results.append(self.conv2(x).view(-1, self.hp.n_filters2))
        conv_results.append(self.conv2(x).view(-1, self.hp.n_filters3))
        x = torch.cat(conv_results, 1)

        logits = {}
        logits['subsite'] = self.fc1(x)
        logits['laterality'] = self.fc2(x)
        logits['behavior'] = self.fc3(x)
        logits['grade'] = self.fc4(x)

        return logits


def new(hyperparameters=None, savefile=None):
    """Create new MTCNN model."""
    if hyperparameters:
        hparams = hyperparameters
    else:
        hparams = Hyperparameters()

    model = MTCNN(hparams)

    if savefile:
        model.load_state_dict(savefile, strict=False)

    return model


def create(hyperparameters=None):
    """ Bayescache factory function """
    def instantiate(**_):
        if hyperparameters:
            hparams = hyperparameters
            print(hparams)
            print(type(hparams))
        else:
            hparams = Hyperparameters()

        #print(type(hparams))
        return MTCNN(hparams)

    return ModelFactory.generic(instantiate)