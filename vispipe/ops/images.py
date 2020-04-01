from ..vispipe import Pipeline
from ..vispipe import block
import cv2
import numpy as np


@block(tag='images')
def random_image(size: tuple = (28, 28)):
    """
    Create a random image of shape specified.

    Parameters
    ----------
    size : tuple
        The size of the random image.
    """
    return np.concatenate([np.random.randint(0, 255, size=size + (3,)), np.ones(size + (1,)) * 255], axis=-1)


@block(tag='images')
def image_rgb(r, g, b, size: tuple = (28, 28)):
    """
    Create an image with fixed color for r, g, b input channels (0 to 255).

    Parameters
    ----------
    size : tuple
        The size of the image you want.
    """
    ones = np.ones(size + (1,))
    return np.concatenate([r * ones, g * ones, b * ones, ones * 255], axis=-1)


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

    returns
    ------
    The image resized to specified resolution.
    """
    return cv2.resize(image, (width, height))


@block(tag='images')
def change_contrast(image, contrast: float = 1.0):
    """
    Change the constrast of an image.

    Parameters
    ----------
    contrast : float
        The contrast you want to set. [0; 1.0] recommended.
    """
    return cv2.convertScaleAbs(image, alpha=contrast, beta=0)


@block(tag='images')
def change_brightness(image, brightness: float = 50):
    """
    Change the brightness of an image.

    Parameters
    ----------
    brightness : float
        The brightness to set.
    """
    return cv2.convertScaleAbs(image, alpha=1.0, beta=brightness)
