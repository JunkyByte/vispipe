from ..vispipe import Pipeline
from ..vispipe import block
import cv2
import numpy as np


@block
def random_image(size: tuple = (28, 28)):
    return np.concatenate([np.random.randint(0, 255, size=size + (3,)), np.ones(size + (1,)) * 255], axis=-1)


@block
def image_rgb(r, g, b, size: tuple = (28, 28)):
    ones = np.ones(size + (1,))
    yield np.concatenate([r * ones, g * ones, b * ones, ones * 255], axis=-1)


@block(tag='images')
def resize_cv2(image, width: int = 224, height: int = 224):
    """
    Resize input image to resolution.

    Parameters
    ----------
    width : int
        The width of the output image.
    height : int
        The height of the output image.

    Yields
    ------
    The image resized to specified resolution.
    """
    yield cv2.resize(image, (width, height))


@block(tag='images')
def change_contrast(image, contrast: float = 1.0):
    yield cv2.convertScaleAbs(image, alpha=contrast, beta=0)


@block(tag='images')
def change_brightness(image, brightness: float = 50):
    yield cv2.convertScaleAbs(image, alpha=1.0, beta=brightness)
