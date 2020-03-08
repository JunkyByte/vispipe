from vispipe import Pipeline
from vispipe import block
"""
Flow generators allow to modify the way a stream of your pipeline is processed.
"""


@block(tag='flows')
class iterator:
    """
    Iterates over its input, will not accept new inputs until it reaches a StopIteration.

    Parameters
    ----------
    return_stop : bool
        if True returns StopIteration as last element of each iteration
        if False skips the StopIteration starting the subsequent iterator directly.

    Yields
    ------
        Elements of the iterable object you passed as input
    """

    def __init__(self, return_stop: bool = False):
        self.iterator = None
        self.return_stop = StopIteration if return_stop else Pipeline._empty

    def run(self, x):
        if self.iterator is None:
            self.iterator = iter(x)

        try:
            y = Pipeline._skip(next(self.iterator))
        except StopIteration:
            self.iterator = None
            y = self.return_stop
        yield y


@block(tag='flows')
class batchify:
    """
    Concatenates subsequent inputs together until it reaches the size requested.

    Parameters
    ----------
    size : int
        The size of each bach.

    Yields
    ------
        Batch of inputs of the size specified
    """

    def __init__(self, size: int = 2):
        self.buffer = []
        self.size = size

    def run(self, x):
        self.buffer.append(x)
        if len(self.buffer) == self.size:
            x = self.buffer
            self.buffer = []
            yield x
        yield Pipeline._empty
