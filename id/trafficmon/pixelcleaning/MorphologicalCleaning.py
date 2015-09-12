from PixelCleaningAbstract import PixelCleaningAbstract
import cv2
import numpy as np

__author__ = 'Luqman'


class MorphologicalCleaning(PixelCleaningAbstract):
    def __init__(self):
        PixelCleaningAbstract.__init__(self, "MorphologicalCleaning")

    def apply(self, image):
        # image is binary image
        # out1 = cv2.erode(image, np.ones((2, 2), np.uint8), iterations=2)
        # out2 = cv2.dilate(out1, np.ones((2, 2), np.uint8), iterations=2)
        # out = cv2.erode(out2, np.ones((2, 2), np.uint8), iterations=1)
        se = np.ones((4, 4), dtype='uint8')
        image_open = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)

        return image_open