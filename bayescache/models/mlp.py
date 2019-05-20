import torch
import torch.nn as nn
import torch.nn.functional as F

from bayescache.metrics.loss_metric import Loss
from bayescache.metrics.accuracy import Accuracy
from bayescache.api import SupervisedModel, ModelFactory


class Hyperparameters:
    n_hidden1 = 100
    n_hidden2 = 100


class MLP(SupervisedModel):

    def __init__(self, hparams, input_size, n_classes=10)
        super(MLP, self).__init__()
        self.hp = hparams
        self.fc1 = nn.Linear(input_size, self.hp.n_hidden1)
        self.fc2 = nn.Linear(self.hp.n_hidden1, self.hp.n_hidden2)
        self.fc3 = nn.Linear(self.hp.n_hidden2, n_classes)


    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        return F.cross_entropy(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def new(input_size, n_classes=10, hyperparameters=None, savefile=None):
    """Create new  multi-layer perceptron."""
    if hyperparameters:
        hparams = hyperparameters
    else:
        hparams = Hyperparameters()

    model = MLP(hparams, input_size, n_classes)

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
        return MLP(hparams)

    return ModelFactory.generic(instantiate)
