from vispipe import Pipeline
from vispipe import block
import numpy as np


@block(tag='common')
def reshape(x, shape: tuple = (28, 28)):
    yield np.reshape(x, shape)


@block(tag='common')
def numpy_cast(x, np_type: str = 'float32'):
    """
    Cast input to any numpy type.

    Parameters
    ----------
    np_type : str
        The numpy type you want to cast to.

    Yields
    ------
        Its input as a numpy array of type specified.
    """
    yield np.array(x).astype(np_type)
