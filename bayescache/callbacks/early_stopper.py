from bayescache.api import BatchInfo, EpochInfo, TrainingInfo, Callback


class EarlyStopper(Callback):
    """Early stopping, tracking validation performance.

    Parameters
    ----------
    patience : int
      Number of epochs to wait while performance does not improve.

    delta : dict
      The difference
    """
    def __init__(self, patience: int, delta: dict, minimize: bool=True):
        self.delta = delta
        self.patience = patience
        self.minimize = minimize

    def on_initialization(self, training_info: TrainingInfo):
        training_info['stopped_epoch'] = 0

    def on_train_begin(self, training_info: TrainingInfo):
        self.wait = 0

    def on_epoch_begin(self, epoch_info: EpochInfo):
        self.metrics_start = epoch_info.result_accumulator.value()

    def on_epoch_end(self, epoch_info: EpochInfo):
        self.metrics_end = epoch_info.result_accumulator.value()

        difference = {}
        for key, value in self.metrics_start:
            difference[key] = self.metrics_start[key] - self.metrics_end[key]

        # TODO: if the majority of metrics do not improve, increase wait.
        if self.minimize:
            pass

    def write_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        hidden_state_dict['stopped_epoch'] = training_info['time']

    def load_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        training_info['time'] = hidden_state_dict['time_tracker/time']


class LossEarlyStopper(Callback):
    """Early stopping, tracking validation loss."""
    def __init__(self, patience: int, delta: float):
        self.delta = delta
        self.patience = patience

    def on_initialization(self, training_info: TrainingInfo):
        training_info['stopped_epoch'] = 0

    def on_train_begin(self, training_info: TrainingInfo):
        self.wait = 0

    def on_epoch_begin(self, epoch_info: EpochInfo):
        self.metrics_start = epoch_info.result_accumulator.value()

    def on_epoch_end(self, epoch_info: EpochInfo):
        self.metrics_end = epoch_info.result_accumulator.value()

        difference = {}
        for key, value in self.metrics_start:
            difference[key] = self.metrics_start[key] - self.metrics_end[key]

        # TODO: if the majority of metrics do not improve, increase wait.
        if self.minimize:
            pass

    def write_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        hidden_state_dict['stopped_epoch'] = training_info['time']

    def load_state_dict(self, training_info: TrainingInfo, hidden_state_dict: dict):
        training_info['time'] = hidden_state_dict['time_tracker/time']