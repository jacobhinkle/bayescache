import numpy as np
import pandas as pd


class TrainingHistory:
    """
    Simple aggregator for the training history.
    An output of training storing scalar metrics in a pandas dataframe.
    """
    def __init__(self):
        self.data = []

    def add(self, epoch_result):
        """ Add a datapoint to the history """
        self.data.append(epoch_result)

    def frame(self):
        """ Return history dataframe """
        return pd.DataFrame(self.data).set_index('epoch_idx')
