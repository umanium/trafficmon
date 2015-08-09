from BackgroundSubtractionAbstract import BackgroundSubtractionAbstract
import numpy as np
import cv2

__author__ = 'Luqman'


class SingleGaussian(BackgroundSubtractionAbstract):

    def __init__(self, image):
        BackgroundSubtractionAbstract.__init__(self, "SingleGaussian")
        self.background_model = BackgroundModel(image)

    def apply(self, image):
        foreground_mask = self.background_model.get_foreground_mask(image, 60)
        self.background_model.update(image, 0.025)
        return foreground_mask


class BackgroundModel(object):
    PI = 3.14
    mean = None
    variance = None

    def __init__(self, image):
        self.mean = np.copy(image).astype(np.float32)
        self.variance = np.ones_like(image) * 6.

    def pdf(self, image):
        lower_value = np.multiply(np.sqrt(self.variance), np.sqrt(2*self.PI))
        upper_value = np.exp((-((image - self.mean) ** 2)) / (2. * self.variance))
        return upper_value / lower_value

    def update(self, image, alpha):

        # update variance
        new_variance = ((1-alpha) * self.variance) + (alpha * ((image - self.mean) ** 2))
        self.variance = new_variance

        # update mean
        new_mean = ((1-alpha) * self.mean) + (alpha * image)
        self.mean = new_mean

    def get_foreground_mask(self, image, threshold):
        # distance_metric = np.sqrt(((image - self.mean) ** 2) / self.variance)  # mahalanobis distance
        threshold_array = np.sqrt(np.ones_like(image).astype(np.float32) * threshold)
        distance_metric = np.abs(np.subtract(image, self.mean))
        foreground_mask = np.where(np.less(distance_metric, threshold_array), 0, 255)
        return cv2.convertScaleAbs(foreground_mask)