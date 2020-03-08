from vispipe import Pipeline
from vispipe import block
import numpy as np


@block(tag='common')
def reshape(x, shape: tuple = (28, 28)):
    yield np.reshape(x, shape)


@block(tag='common')
def cast_to_float32(x):
    yield np.array(x, dtype=np.float32)


@block(tag='common')
def cast_to_uint8(x):
    yield np.array(x, dtype=np.uint8)
