from vispipe.vispipe import block
import numpy as np
"""
Input generators for your pipeline.
"""


@block(tag='input_file')
class numpy_file:
    """
    Yields the raw buffer, line by line from a numpy file.
    This is equivalent to stacking a numpy_flow with a constant input into an iterator

    Parameters
    ----------
    path : str
        The path to the file you want to load.


    Yields
    ------
        The Content of the loaded array line by line.
    """

    def __init__(self):
        self.file = None

    def run(self, path: str = ''):
        if self.file is None:
            self.file = iter(np.load(path))
        yield next(self.file)


@block(tag='input_flow', max_queue=1)
def numpy_flow(path):
    """
    Yields the raw buffer from the path provided as input.

    Yields
    ------
        The full content of the loaded array from the path you provided.
    """
    yield np.load(path)
