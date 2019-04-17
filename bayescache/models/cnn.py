import torch
import torch.nn as nn
import torch.nn.functional as F

from bayescache.metrics.loss_metric import Loss
from bayescache.metrics.accuracy import Accuracy
from bayescache.api import SupervisedModel, ModelFactory


class Hyperparameters:
    kernel1 = 5
    kernel2 = 5 


class CNN(SupervisedModel):

    def __init__(self, hparams, num_classes):
        super(CNN, self).__init__()
        self.hp = hparams
        self.conv1 = nn.Conv2d(1, 20, self.hp.kernel1, 1)
        self.conv2 = nn.Conv2d(20, 50, self.hp.kernel2, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        return F.cross_entropy(y_pred, y_true)

    def metrics(self):
        """ Set of metrics for this model """
        return [Loss(), Accuracy()]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def new(hyperparameters=None, num_classes=10, savefile=None):
    """Create new MTCNN model."""
    if hyperparameters:
        hparams = hyperparameters
    else:
        hparams = Hyperparameters()

    model = CNN(hparams, num_classes)

    if savefile:
        model.load_state_dict(savefile, strict=False)

    return model


def create(hyperparameters=None, num_classes=10):
    """ Vel factory function """
    def instantiate(**_):
        if hyperparameters:
            hparams = hyperparameters
        else:
            hparams = Hyperparameters()

        return CNN(hparams, num_classes)

    return ModelFactory.generic(instantiate)