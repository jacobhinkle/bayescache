class LossMeter:
    """Record training and validation loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.train_loss = []
        self.valid_loss = []

    def add_train_loss(self, loss):
        self.train_loss.append(loss)

    def add_valid_loss(self, loss):
        self.valid_loss.append(loss)

    def get_train_loss(self):
        return self.train_loss

    def get_valid_loss(self):
        return self.valid_loss
