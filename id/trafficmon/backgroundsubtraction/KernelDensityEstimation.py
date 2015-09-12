from BackgroundSubtractionAbstract import BackgroundSubtractionAbstract
import numpy as np

__author__ = 'Luqman'


class KernelDensityEstimation(BackgroundSubtractionAbstract):
    background_model = None

    def __init__(self, image, m):
        BackgroundSubtractionAbstract.__init__(self, "KernelDensityEstimation")
        self.background_model = BackgroundModel(image, m)

    def apply(self, image):
        density_estimation = self.background_model.density_estimation(image)
        self.background_model.update_kernel(image)
        return image

    def get_background_model(self):
        return self.background_model


class BackgroundModel(object):
    kernels = []
    PI = 3.14
    counter = 0

    def __init__(self, image, m):
        self.kernels.append(image)
        self.variance = np.ones_like(image)
        self.mean = np.ones_like(image)
        self.m = m
        self.pdf_dict = self.pdf_dict = np.array([self.pdf(i) for i in range(256)])

    # def pdf(self, image):
    #     lower_value = np.multiply(np.sqrt(self.variance), np.sqrt(2*self.PI))
    #     upper_value = np.exp((-((image - self.mean) ** 2)) / (2. * self.variance))
    #     return upper_value / lower_value

    def pdf(self, num):
        variance_single = 100
        index = -(float(num) ** 2) / (2 * variance_single)
        result = np.exp(index) / np.sqrt(2 * self.PI * variance_single)
        return result

    def density_estimation(self, image):
        pdf_kernel = map(
            lambda x: np.multiply(
                np.ones_like(image, 'float32'),
                self.pdf_dict[np.absolute(np.subtract(image, x))]
            ),
            self.kernels
        )

        # pdf_kernel = self.kernels
        pdf_all = reduce(lambda a, b: np.add(a, b), pdf_kernel)
        return pdf_all

    def get_density_range(self, image):
        density_estimation = self.density_estimation(image)
        density_estimation_flat = density_estimation.flatten()
        return density_estimation_flat.min(), density_estimation_flat.max(), sum(density_estimation_flat)/len(density_estimation_flat)

    def is_kernel_full(self):
        return len(self.kernels) == self.m

    def update_kernel(self, image):
        if not self.is_kernel_full():
            self.kernels.append(image)
        else:
            self.kernels[self.counter % self.m] = image
        self.counter += 1