from BackgroundSubtractionAbstract import BackgroundSubtractionAbstract
import numpy as np

__author__ = 'Luqman'


class MixtureOfGaussian(BackgroundSubtractionAbstract):
    def __init__(self, n, image):
        BackgroundSubtractionAbstract.__init__(self, "MixtureOfGaussian")
        self.background_model = [GaussianModel(image, i) for i in range(n)]

    def update_model(self, image):
        pass

    def classify(self, image):
        pass


class GaussianModel(object):
    PI = 3.14
    mean = None
    amplitude = None
    variance = None
    index = 0

    def __init__(self, image, idx):
        self.index = idx
        self.mean = np.ones_like(image, np.float32)
        self.amplitude = np.ones_like(image, np.float32)
        self.variance = np.ones_like(image, np.float32)

    def pdf(self, image):
        lower_value = np.multiply(np.sqrt(self.variance), np.sqrt(2*self.PI))
        upper_value = np.exp((-((image - self.mean) ** 2)) / (2. * self.variance))
        return upper_value / lower_value

    def update(self, image, foreground_mask, alpha):
        # p
        p = alpha * self.pdf(image)

        # update amplitude
        new_amplitude = ((1-alpha) * self.amplitude) + (alpha * foreground_mask)
        self.amplitude = np.copy(new_amplitude)

        # update mean
        new_mean = ((1-p) * self.mean) + (p * image)
        self.mean = new_mean

        # update variance
        new_variance = ((1-p) * self.variance) + (p * ((image - self.mean) ** 2))
        self.variance = new_variance

    def get_foreground_mask(self, image):  # using stauffer-grimson

        range_up = np.add(self.mean, (2.5 * np.sqrt(self.variance)))
        range_down = np.subtract(self.mean, (2.5 * np.sqrt(self.variance)))
        foreground_mask = np.where(
            np.logical_and(np.greater_equal(image, range_down), np.less_equal(image, range_up)),
            1,
            0
        )
        return foreground_mask
