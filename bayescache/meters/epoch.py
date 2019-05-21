class EpochMeter:
    """Count epochs"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.n = 0

    def increment(self):
        self.n += 1

    def get_counts(self):
        return self.n
