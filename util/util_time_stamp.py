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
        self._diff = 0
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
        self._diff = self.end - self.start
        m, s = divmod(self._diff, 60)
        h, m = divmod(m, 60)
        self.diff = "%02d:%02d:%02.2f" % (h, m, s)
        if iters:
            self.diff_average = self._diff / iters

        m, s = divmod(time.time() - self.init, 60)
        h, m = divmod(m, 60)
        self.from_begin = "%02d:%02d:%02d" % (h, m, s)
