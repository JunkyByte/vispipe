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
    yield value


@block(tag='common')
def constant(value: float = 0):
    """
    A constant float.

    Parameters
    ----------
    value : float
        The float value.
    """
    yield value


@block(tag='common')
def index(x, index: int = 0):
    """
    Yields its input indexed by index value.

    [TODO:description]

    Parameters
    ----------
    index : int
        The index value.
    """
    yield x[index]


@block(tag='common')
def key(x, key: str = ''):
    """
    Yields its input indexed by key

    Parameters
    ----------
    key : str
        The key value.
    """
    yield x[key]


@block(tag='common')
def index_slice(x, slice_value: slice = ''):
    """
    Yield its input sliced by value

    Parameters
    ----------
    slice_value : slice
        The slice value.
    """
    yield x[slice_value]


@block(tag='common')
def np_cast(x, np_type: str = 'float32'):
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


@block(tag='common')
def np_reshape(x, shape: tuple = (28, 28)):
    """
    Reshape its input to shape value

    Parameters
    ----------
    shape : tuple
        The shape value
    """
    yield np.reshape(x, shape)


@block(tag='common')
def np_linspace(start: float = 0, stop: float = 0, num: int = 50, endpoint: bool = True,
        retstep: bool = False, dtype: np.dtype = None):
    """
    Yields a numpy linear space.
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
    yield np.linspace(start, stop, num, endpoint, retstep, dtype)


@block(tag='common')
def np_randn(shape: tuple = (1,)):
    """
    Yields random numbers with shape specified from a normal distribution.
    See numpy random randn function for more details.

    Parameters
    ----------
    shape : tuple
        The output shape.
    """
    yield np.random.randn(*shape)


@block(tag='common')
def np_randint(low: int = 0, high: int = None, size: int = None, dtype: str = 'l'):
    """
    Yields random integers with low / high and size specified.
    Refer to numpy random randint documentation for more details.

    Parameters
    ----------
    low : int
    high : int
    size : int
    dtype : str
    """
    yield np.random.randint(low, high, size, dtype)


@block(tag='common')
def np_ones(shape: tuple = None, dtype: np.dtype = None, order: str = 'C'):
    """
    Yields a numpy array of the shape specified filled with ones.
    Refer to numpy documentation for more details.

    Parameters
    ----------
    shape : tuple
    dtype : np.dtype
    order : str
    """
    yield np_ones(shape, dtype, order)


@block(tag='common')
def np_zeros(shape: tuple = None, dtype: np.dtype = None, order: str = 'C'):
    """
    Yields a numpy array of the shape specified filled with zeros.
    Refer to numpy documentation for more details.

    Parameters
    ----------
    shape : tuple
    dtype : np.dtype
    order : str
    """
    yield np.zeros(shape, dtype, order)


@block(tag='common')
def np_empty(shape: tuple = None, dtype: np.dtype = None, order: str = 'C'):
    """
    Yields an empty numpy array of the shape specified.
    Refer to numpy documentation for more details.

    Parameters
    ----------
    shape : tuple
    dtype : np.dtype
    order : str
    """
    yield np.empty(shape, dtype, order)


@block(tag='common', intercept_end=True)
class accumulate:
    """
    Yields the summed value of its inputs every time it receives a StopIteration.
    """
    def __init__(self):
        self.sum = 0

    def run(self, x):
        if x is StopIteration:
            yield self.sum
        self.sum += x
        yield Pipeline._empty


@block(tag='common')
def filter_func(x, func: Callable = lambda x: True):
    """
    Yields its inputs filtered by func function.

    Parameters
    ----------
    func : Callable
        The function you want to filter with, should return a boolean.
    """
    yield x if func(x) else Pipeline._empty


@block(tag='common')
def add(x, y):
    """
    """
    yield x + y


@block(tag='common')
def subtract(x, y):
    """
    """
    yield x - y


@block(tag='common')
def multiply(x, y):
    """
    """
    yield x * y


@block(tag='common')
def divide(x, y):
    """
    """
    yield x / y


@block(tag='common')
def integer_division(x, y):
    """
    x // y
    """
    yield x // y


@block(tag='common')
def modulo(x, y):
    """
    x % y
    """
    yield x % y


@block(tag='common')
def sin(x):
    """
    """
    yield math.sin(x)


@block(tag='common')
def cos(x):
    """
    """
    yield math.cos(x)


@block(tag='common')
def tan(x):
    """
    """
    yield math.tan(x)


@block(tag='common')
def asin(x):
    """
    """
    yield math.asin(x)


@block(tag='common')
def acos(x):
    """
    """
    yield math.acos(x)


@block(tag='common')
def atan(x):
    """
    """
    yield math.atan(x)


@block(tag='common')
def sinh(x):
    """
    """
    yield math.sinh(x)


@block(tag='common')
def cosh(x):
    """
    """
    yield math.cosh(x)


@block(tag='common')
def tanh(x):
    """
    """
    yield math.tanh(x)
