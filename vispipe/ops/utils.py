from ..vispipe import Pipeline
from ..vispipe import block
import time
import logging


@block(tag='utils', max_queue=100)
class benchmark:
    """
    Benchmark the flow of the pipeline up to this point.
    Will yield the result once it reaches the number of iteration requested.

    Parameters
    ----------
    n : int
        The number of outputs the pipeline will pass to this block before it outputs.
        the statistics.
    end : bool
        Whether to kill the pipeline once it finished the first benchmark.
    log : bool
        Whether to log the message as well.
    """
    def __init__(self, n: int = 1000, end: bool = True, log: bool = False):
        self.n = n
        self.start_n = self.n
        self.started = False
        self.end = end
        self.log = log
        self.finished = False

    def run(self, x):
        if not self.started:  # Start the timer
            self.started = True
            self.start_time = time.time()

        self.n -= 1
        if self.n == -1:  # If we reached the iterations
            delta = time.time() - self.start_time
            msg = 'Benchmark - %s runs | time: %s | run/s: %s' % (self.start_n, delta, round(self.start_n / delta, 4))
            if self.log:
                logging.getLogger('vispipe').info(msg)

            if self.end:  # Prepare to end
                self.finished = True
            else:  # Restart the timer
                self.n = self.start_n
                self.start_time = time.time()
            yield msg

        if self.n < -1:
            raise StopIteration

        yield Pipeline._empty
