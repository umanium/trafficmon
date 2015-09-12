from BackgroundSubtractionAbstract import BackgroundSubtractionAbstract
import numpy as np
import cv2

__author__ = 'Luqman'


class KMeans(BackgroundSubtractionAbstract):

    def __init__(self, image, k):
        BackgroundSubtractionAbstract.__init__(self, "KMeans")
        self.background_model = BackgroundModel(image, k, 0.02)

    def apply(self, image):
        centroid_diff = self.background_model.get_diff(image)
        min_diff_index = np.argmin(centroid_diff, axis=0)
        weight = self.background_model.get_weight(min_diff_index)
        fg = np.where(np.greater(weight, self.background_model.get_threshold()), 0, 255)
        self.background_model.update(min_diff_index, image)
        return cv2.convertScaleAbs(fg)

    def get_background_model(self):
        return self.background_model


class BackgroundModel(object):
    w = None
    centroids = None
    k = 0
    alpha = 0.02

    def __init__(self, image, k, alpha):
        self.centroids = [np.multiply(np.ones_like(image, 'float64'), ((256. / k) * i)) for i in range(k)]
        self.w = [np.multiply(np.ones_like(image, 'float64'), (1. / k))] * k
        self.k = k
        self.alpha = alpha
        return

    def get_diff(self, image):
        diff = map(
            lambda x: np.absolute(np.subtract(image, x)),
            self.centroids
        )
        return diff

    def get_weight(self, min_diff_index):
        weight = np.zeros_like(min_diff_index)
        for i in range(self.k):
            weight = np.where(np.equal(min_diff_index, i), self.w[i], weight)
        return weight

    def update(self, min_diff_index, image):

        # update centroids and weight
        updated_centroid = \
            [np.where(
                np.equal(min_diff_index, idx),
                (np.add(centroid, self.alpha * np.subtract(image, centroid))),
                centroid
            ) for idx, centroid in enumerate(self.centroids)]
        updated_weight = \
            [np.where(
                np.equal(min_diff_index, idx),
                np.add(np.multiply((1. - self.alpha), w), self.alpha),
                np.multiply((1. - self.alpha), w)
            ) for idx, w in enumerate(self.w)]

        self.centroids = updated_centroid
        self.w = updated_weight

    def get_threshold(self):
        return 1./self.k
