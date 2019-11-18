"""Count the time while training."""
import time


class Time:
    """Timestamp."""

    def __init__(self):
        """Init."""
        self.init = time.time()
        self.start = 0
        self.end = 0
        self.stamp_diff = 0
        self.diff = 0
        self.diff_average = 0
        self.from_begin = 0

    def time_start(self):
        """Start time."""
        self.start = time.time()

    def time_stamp(self):
        """Time stamp."""
        self.stamp_diff = time.time() - self.start

    def time_end(self, iters=0):
        """End time."""
        self.end = time.time()
        self.diff = self.end - self.start
        if iters:
            self.diff_average = self.diff / iters
        self.from_begin = time.time() - self.init
