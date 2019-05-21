import time


class TimeMeter:
    """Measure time"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.elapsed_time = 0.0
        self.time = time.time()

    def stop_timer(self):
        self.elapsed_time = time.time() - self.time

    def get_timings(self):
        return self.elapsed_time