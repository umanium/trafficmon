from BackgroundSubtractionAbstract import BackgroundSubtractionAbstract
import numpy as np
import cv2

__author__ = 'Luqman'


class KalmanFilter(BackgroundSubtractionAbstract):

    def __init__(self, image, th):
        BackgroundSubtractionAbstract.__init__(self, "KalmanFilter")
        self.background_model = BackgroundModel(image, th)

    def apply(self, image):
        fg = self.background_model.apply(image)
        return cv2.convertScaleAbs(fg)

    def get_background_model(self):
        return self.background_model


class BackgroundModel(object):
    b_t = None  # background model, Bt
    b_t1 = None  # temporal derivative, B't
    alpha_1 = 0.1
    alpha_2 = 0.01
    a = [[1., 0.7], [0., 0.7]]  # background_dynamics
    h = [1., 0.]  # measurement_matrix
    threshold = None

    def __init__(self, image, th):
        self.b_t1 = np.copy(image)
        self.b_t = np.zeros_like(image)
        self.threshold = np.ones_like(image) * th
        return

    def apply(self, image):
        # get foreground mask
        fg_mask = np.where(
            np.greater(
                np.absolute(np.subtract(image, self.b_t)),
                self.threshold
            ),
            255,
            0
        )

        # update background (Bt and B't)
        k_t = np.where(np.equal(fg_mask, 255), self.alpha_1, self.alpha_2)
        b_t_dynamics = (self.a[0][0] * self.b_t) + (self.a[0][1] * self.b_t1)
        b_t_measurements = k_t * (image - (self.h[0] * b_t_dynamics))
        b_t1_dynamics = (self.a[1][0] * self.b_t) + (self.a[1][1] * self.b_t1)
        b_t1_measurements = k_t * (image - (self.h[0] * b_t1_dynamics))

        self.b_t = b_t_dynamics + b_t_measurements
        self.b_t1 = b_t1_dynamics + b_t1_measurements

        print b_t_measurements

        return fg_mask