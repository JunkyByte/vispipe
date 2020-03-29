from ..vispipe import Pipeline
from ..vispipe import block
"""
Flow generators allow to modify the way a stream of your pipeline is processed.
"""


@block(tag='flows', intercept_end=True)
class iterator:
    """
    Iterates over its input, will not accept new inputs until it reaches a StopIteration.

    Yields
    ------
        Elements of the iterable object you passed as input.
    """

    def __init__(self):
        self.iterator = None

    def run(self, x):
        if self.iterator is None:
            self.iterator = iter(x)

        try:
            y = Pipeline._skip(next(self.iterator))
        except StopIteration:
            self.iterator = None
            y = Pipeline._empty
        yield y


@block(tag='flows', intercept_end=True)
class batchify:
    """
    Concatenates subsequent inputs together until it reaches the size requested.

    Parameters
    ----------
    size : int
        The size of each bach.

    Yields
    ------
        Batch of inputs of the size specified.
    """

    def __init__(self, size: int = 2):
        self.buffer = []
        self.size = size

    def run(self, x):
        self.buffer.append(x)
        if len(self.buffer) == self.size or x is StopIteration:
            x = self.buffer
            self.buffer = []
            yield x if x is not StopIteration else x[:-1]
        yield Pipeline._empty
