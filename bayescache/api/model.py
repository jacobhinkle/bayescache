import hashlib
import torch
import torch.nn as nn

import vel.util.module_util as mu

from bayescache.metrics.loss_metric import Loss
from vel.util.summary import summary


class Model(nn.Module):
    """ Class representing full neural network model """

    def metrics(self) -> list:
        """ Set of metrics for this model """
        return [Loss()]

    def train(self, mode=True):
        """
        Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Returns:
            Module: self
        """
        super().train(mode)

        if mode:
            mu.apply_leaf(self, mu.set_train_mode)

        return self

    def summary(self, input_size=None, hashsummary=False):
        """ Print a model summary """

        if input_size is None:
            print(self)
            print("-" * 80)
            number = sum(p.numel() for p in self.parameters())
            print("Number of model parameters: {:,}".format(number))
            print("-" * 80)
        else:
            summary(self, input_size)

        if hashsummary:
            for idx, hashvalue in enumerate(self.hashsummary()):
                print(f"{idx}: {hashvalue}")

    def hashsummary(self):
        """ Print a model summary - checksums of each layer parameters """
        children = list(self.children())

        result = []

        for child in children:
            result.extend(hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest() for x in child.parameters())

        return result

    def get_layer_groups(self):
        """ Return layers grouped """
        return [self]

    def reset_weights(self):
        """ Call proper initializers for the weights """
        pass

    @property
    def is_recurrent(self) -> bool:
        """ If the network is recurrent and needs to be fed state as well as the observations """
        return False


class RnnModel(Model):
    """ Class representing recurrent model """

    @property
    def is_recurrent(self) -> bool:
        """ If the network is recurrent and needs to be fed previous state """
        return True

    @property
    def state_dim(self) -> int:
        """ Dimension of model state """
        raise NotImplementedError

    def zero_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, self.state_dim)


class BackboneModel(Model):
    """ Model that serves as a backbone network to connect your heads to """


class RnnLinearBackboneModel(BackboneModel):
    """
    Model that serves as a backbone network to connect your heads to -
    one that spits out a single-dimension output and is a recurrent neural network
    """

    @property
    def is_recurrent(self) -> bool:
        """ If the network is recurrent and needs to be fed previous state """
        return True

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        raise NotImplementedError

    @property
    def state_dim(self) -> int:
        """ Dimension of model state """
        raise NotImplementedError

    def zero_state(self, batch_size):
        """ Initial state of the network """
        return torch.zeros(batch_size, self.state_dim, dtype=torch.float32)


class LinearBackboneModel(BackboneModel):
    """
    Model that serves as a backbone network to connect your heads to - one that spits out a single-dimension output
    """

    @property
    def output_dim(self) -> int:
        """ Final dimension of model output """
        raise NotImplementedError


class SupervisedModel(Model):
    """ Model for a supervised learning problem """
    def loss(self, x_data, y_true):
        """ Forward propagate network and return a value of loss function """
        y_pred = self(x_data)
        return y_pred, self.loss_value(x_data, y_true, y_pred)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        raise NotImplementedError


class MultiTaskSupervisedModel(Model):
    """ Model for a multi-task supervised learning problem.

    This is similar to the SupervisedModel, but where 
    `y_true` and `y_pred` are dictionaries with matching keys.
    MultiTaskSupervised models have the option of reducing the loss 
    over all tasks by either the sum or the mean.
    """
    def loss(self, x_data, y_true, reduce=None):
        """ Forward propagate network and return a value of loss function """
        # TODO: This may need to be moved to the model.
        if reduce not in (None, 'sum', 'mean'):
            raise ValueError("`reduce` must be either None, `sum`, or `mean`!")

        y_pred = self(x_data)
        losses = {}
        for key, value in y_true:
            # TODO: test this bad boy.
            # y_true and y_pred must have the same keys.
            losses[key] = self.loss_value(x_data, y_true[key], F.softmax(y_pred[key]))

        if reduce:
            total = 0
            for _, value in losses.items():
                total += value
            
            if reduce == "mean":
                losses = total / len(losses)
            elif reduce == "sum":
                losses = total

        return y_pred, losses

    def loss_value(self, x_data, y_true, y_pred, reduce=None):
        """ Calculate a value of loss function """
        raise NotImplementedError


class RnnSupervisedModel(RnnModel):
    """ Model for a supervised learning problem """

    def loss(self, x_data, y_true):
        """ Forward propagate network and return a value of loss function """
        y_pred = self(x_data)
        return y_pred, self.loss_value(x_data, y_true, y_pred)

    def loss_value(self, x_data, y_true, y_pred):
        """ Calculate a value of loss function """
        raise NotImplementedError
