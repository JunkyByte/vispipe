from ..vispipe import Pipeline
from ..vispipe import block
import numpy as np


@block(tag='flows', intercept_end=True)
class iterator:
    """
    Iterates over its input, will not accept new inputs until it reaches a StopIteration.

    returns
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
        return y


@block(tag='flows', intercept_end=True)
class batchify:
    """
    Concatenates subsequent inputs together until it reaches the size requested.

    Parameters
    ----------
    size : int
        The size of each bach.
    to_array: bool
        Whether you want the output as a np array or not.

    returns
    ------
        Batch of inputs of the size specified.
    """
    def __init__(self, size: int = 2, to_array: bool = True):
        self.size = size
        self.to_array = to_array
        self.buffer = []

    def run(self, x):
        self.buffer.append(x)
        if len(self.buffer) == self.size or x is StopIteration:
            x = np.array(self.buffer) if self.to_array else self.buffer
            self.buffer = []
            return x if x is not StopIteration else x[:-1]
        return Pipeline._empty
