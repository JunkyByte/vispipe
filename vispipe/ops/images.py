from vispipe import Pipeline
from vispipe import block
import cv2


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
