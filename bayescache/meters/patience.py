class PatienceMeter:
    """Patience for validation improvement"""
    def __init__(self, patience=5):
        self.patience = patience
        self.reset()

    def reset(self):
        self.counter = 0
        self.epoch = 0
        self.minimal_loss = 9e10
        self.stop_early = False

    def check_loss(self, loss):
        self.epoch += 1

        if loss < self.minimal_loss:
            self.minimal_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter > self.patience:
            self.stop_early = True

    def get_stop_epoch(self):
        return self.epoch
