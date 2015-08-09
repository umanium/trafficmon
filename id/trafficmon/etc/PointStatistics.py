import cv2
import matplotlib.pyplot as plt
import numpy as np

__author__ = 'Luqman'


class PointValueScatter(object):
    red_values = None
    green_values = None
    blue_values = None
    gray_values = None

    def __init__(self):
        self.red_values = []
        self.green_values = []
        self.blue_values = []
        self.gray_values = []

    def fetch_point_value(self, image, x, y):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.red_values.append(image[y, x, 2])
        self.green_values.append(image[y, x, 1])
        self.blue_values.append(image[y, x, 0])
        self.gray_values.append(gray_image[y, x])

    def plot_histogram(self):
        plt.hist(self.gray_values, color='gray')
        plt.show()

        plt.hist(self.red_values, color='red')
        plt.hist(self.green_values, color='green')
        plt.hist(self.blue_values, color='blue')
        # plt.plot(self.gray_values)
        plt.show()
        pass
