from ..vispipe import Pipeline
from ..vispipe import block
from typing import Callable
import numpy as np
import math


@block(tag='common')
def string(value: str = ''):
    """
    A constant string.

    Parameters
    ----------
    value : str
        The string value.
    """
    return value


@block(tag='common')
def constant(value: float = 0):
    """
    A constant float.

    Parameters
    ----------
    value : float
        The float value.
    """
    return value


@block(tag='common')
def index(x, index: int = 0):
    """
    returns its input indexed by index value.

    Parameters
    ----------
    index : int
        The index value.
    """
    return x[index]


@block(tag='common')
def key(x, key: str = ''):
    """
    returns its input indexed by key

    Parameters
    ----------
    key : str
        The key value.
    """
    return x[key]


@block(tag='common')
def index_slice(x, slice_value: slice = ''):
    """
    return its input sliced by value

    Parameters
    ----------
    slice_value : slice
        The slice value.
    """
    return x[slice_value]


@block(tag='common')
def np_cast(x, np_type: str = 'float32'):
    """
    Cast input to any numpy type.

    Parameters
    ----------
    np_type : str
        The numpy type you want to cast to.

    returns
    ------
        Its input as a numpy array of type specified.
    """
    return np.array(x).astype(np_type)


@block(tag='common')
def np_reshape(x, shape: tuple = (28, 28)):
    """
    Reshape its input to shape value

    Parameters
    ----------
    shape : tuple
        The shape value
    """
    return np.reshape(x, shape)


@block(tag='common')
def np_linspace(start: float = 0, stop: float = 0, num: int = 50, endpoint: bool = True,
        retstep: bool = False, dtype: np.dtype = None):
    """
    returns a numpy linear space.
    Refer to numpy linspace function for argument description

    Parameters
    ----------
    start : float
    stop : float
    num : int
    endpoint : bool
    retstep : bool
    dtype : np.dtype
    """
    return np.linspace(start, stop, num, endpoint, retstep, dtype)


@block(tag='common')
def np_randn(shape: tuple = (1,)):
    """
    returns random numbers with shape specified from a normal distribution.
    See numpy random randn function for more details.

    Parameters
    ----------
    shape : tuple
        The output shape.
    """
    return np.random.randn(*shape)


@block(tag='common')
def np_randint(low: int = 0, high: int = None, size: int = None, dtype: str = 'l'):
    """
    returns random integers with low / high and size specified.
    Refer to numpy random randint documentation for more details.

    Parameters
    ----------
    low : int
    high : int
    size : int
    dtype : str
    """
    return np.random.randint(low, high, size, dtype)


@block(tag='common')
def np_ones(shape: tuple = None, dtype: np.dtype = None, order: str = 'C'):
    """
    returns a numpy array of the shape specified filled with ones.
    Refer to numpy documentation for more details.

    Parameters
    ----------
    shape : tuple
    dtype : np.dtype
    order : str
    """
    return np_ones(shape, dtype, order)


@block(tag='common')
def np_zeros(shape: tuple = None, dtype: np.dtype = None, order: str = 'C'):
    """
    returns a numpy array of the shape specified filled with zeros.
    Refer to numpy documentation for more details.

    Parameters
    ----------
    shape : tuple
    dtype : np.dtype
    order : str
    """
    return np.zeros(shape, dtype, order)


@block(tag='common')
def np_empty(shape: tuple = None, dtype: np.dtype = None, order: str = 'C'):
    """
    returns an empty numpy array of the shape specified.
    Refer to numpy documentation for more details.

    Parameters
    ----------
    shape : tuple
    dtype : np.dtype
    order : str
    """
    return np.empty(shape, dtype, order)


@block(tag='common', intercept_end=True)
class accumulate:
    """
    returns the summed value of its inputs every time it receives a StopIteration.
    """
    def __init__(self):
        self.sum = 0

    def run(self, x):
        if x is StopIteration:
            return self.sum
        self.sum += x
        return Pipeline._empty


@block(tag='common')
def filter_func(x, func: Callable = lambda x: True):
    """
    returns its inputs filtered by func function.

    Parameters
    ----------
    func : Callable
        The function you want to filter with, should return a boolean.
    """
    return x if func(x) else Pipeline._empty


@block(tag='common')
def add(x, y):
    """
    """
    return x + y


@block(tag='common')
def subtract(x, y):
    """
    """
    return x - y


@block(tag='common')
def multiply(x, y):
    """
    """
    return x * y


@block(tag='common')
def divide(x, y):
    """
    """
    return x / y


@block(tag='common')
def integer_division(x, y):
    """
    x // y
    """
    return x // y


@block(tag='common')
def modulo(x, y):
    """
    x % y
    """
    return x % y


@block(tag='common')
def sin(x):
    """
    """
    return math.sin(x)


@block(tag='common')
def cos(x):
    """
    """
    return math.cos(x)


@block(tag='common')
def tan(x):
    """
    """
    return math.tan(x)


@block(tag='common')
def asin(x):
    """
    """
    return math.asin(x)


@block(tag='common')
def acos(x):
    """
    """
    return math.acos(x)


@block(tag='common')
def atan(x):
    """
    """
    return math.atan(x)


@block(tag='common')
def sinh(x):
    """
    """
    return math.sinh(x)


@block(tag='common')
def cosh(x):
    """
    """
    return math.cosh(x)


@block(tag='common')
def tanh(x):
    """
    """
    return math.tanh(x)
