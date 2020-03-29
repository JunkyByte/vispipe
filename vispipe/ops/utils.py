from vispipe import Pipeline
from vispipe import block
import time
import logging


@block(tag='utils', max_queue=100)
class benchmark:
    def __init__(self, n: int = 1000, end: bool = True, log: bool = False):
        self.n = n
        self.start_n = self.n
        self.started = False
        self.end = end
        self.log = log
        self.last_result = 'Counting'

    def run(self, x):
        if not self.started:
            self.started = True
            self.start_time = time.time()
        self.n -= 1
        if self.n == -1:
            end_time = time.time()
            delta = end_time - self.start_time
            self.last_result = 'Benchmark - %s runs | time: %s | r/s: %s' % (self.start_n, delta, round(self.start_n / delta, 4))
            if self.log:
                logging.getLogger('vispipe').info(self.last_result)
            if self.end:
                raise StopIteration
            self.n = self.start_n
            self.start_time = time.time()
        yield self.last_result
