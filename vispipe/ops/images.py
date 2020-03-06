from vispipe.vispipe import Pipeline
from vispipe.vispipe import block
import cv2


@block(tag='images')
def resize_cv2(image, width: int = 224, height: int = 224):
    yield cv2.resize(image, (width, height))
