from vispipe import Pipeline
from vispipe import block
import numpy as np


@block(tag='common')
def constant(value: int = 0):
    yield value


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
    yield np.reshape(x, shape)


@block(tag='common')
def np_linspace(start: float = 0, stop: float = 0, num: int = 50, endpoint: bool = True,
        retstep: bool = False, dtype: np.dtype = None):
    yield np.linspace(start, stop, num, endpoint, retstep, dtype)


@block(tag='common')
def np_randn(shape: tuple = (1,)):
    yield np.random.randn(*shape)


@block(tag='common')
def np_randint(low: int = 0, high: int = None, size: int = None, dtype: str = 'l'):
    yield np.random.randint(low, high, size, dtype)


@block(tag='common')
def np_ones(shape: tuple = None, dtype: np.dtype = None, order: str = 'C'):
    yield np_ones(shape, dtype, order)


@block(tag='common')
def np_zeros(shape: tuple = None, dtype: np.dtype = None, order: str = 'C'):
    yield np.zeros(shape, dtype, order)


@block(tag='common')
def np_empty(shape: tuple = None, dtype: np.dtype = None, order: str = 'C'):
    yield np.empty(shape, dtype, order)


@block(tag='common', intercept_end=True, output_names=[])
class accumulate:
    def __init__(self):
        self.sum = 0

    def run(self, x):
        if x is StopIteration:
            yield self.sum
        self.sum += x
        yield Pipeline._empty


@block(tag='common')
def add(x, y):
    yield x + y


@block(tag='common')
def subtract(x, y):
    yield x - y


@block(tag='common')
def multiply(x, y):
    yield x * y


@block(tag='common')
def divide(x, y):
    yield x / y


@block(tag='common')
def integer_division(x, y):
    yield x // y


@block(tag='common')
def modulo(x, y):
    yield x % y
