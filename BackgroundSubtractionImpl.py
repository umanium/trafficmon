__author__ = 'Luqman'

import cv2
import numpy as np
import PostProcessing
from BackgroundSubtraction import BackgroundSubtraction
from BackgroundSubtraction import BackgroundSubtractionColor

"""
Author: Luqman A. M.
BackgroundSubtractionImpl.py
Background Subtraction Algorithms Implementation for Object Detection in Video Processing
Frame Difference, Running Average, Median, Online K-Means, 1-G, KDE
"""


# class Frame Difference
class FrameDifference(BackgroundSubtraction):
    def __init__(self, filename, threshold):
        print "initializing Frame Difference..."
        BackgroundSubtraction.__init__(self, filename, False)
        self.init_threshold = threshold
        self.threshold = None
        return

    def apply(self, data):
        if self.threshold is None:
            self.threshold = np.multiply(np.ones_like(data, 'uint8'), self.init_threshold)
        diff = np.absolute(np.subtract(data, self.prev_frame))
        new_fg = np.multiply(
            np.ones_like(data, 'uint8'),
            np.where(
                np.less(diff, self.threshold),
                0,
                255
            )
        )
        return new_fg

    def run(self):
        BackgroundSubtraction.run(self)


# class Running Average
class RunningAverage(BackgroundSubtraction):
    def __init__(self, filename, alpha):
        print "initializing Running Average..."
        BackgroundSubtraction.__init__(self, filename, True)
        self.alpha = alpha
        self.beta = 0.02
        return

    def apply(self, data):
        rects, fg = PostProcessing.foreground_detection(data, self.bg, False)
        new_bg = np.where(
            np.equal(fg, 0)
            , np.add(((1 - self.alpha) * self.bg), (self.alpha * data))
            , np.add(((1 - self.beta) * self.bg), (self.beta * data))
        )
        return cv2.convertScaleAbs(new_bg)

    def run(self):
        BackgroundSubtraction.run(self)


# class Running Average 2 (with improvement)
class RunningAverageWithThresholdImprovement(BackgroundSubtraction):
    def __init__(self, filename, alpha):
        print "initializing Running Average..."
        BackgroundSubtraction.__init__(self, filename, False)
        self.alpha = alpha
        self.beta = 0.02
        self.gamma = 3.2
        self.threshold = None
        return

    def apply(self, pict):
        if self.threshold is None:
            self.threshold = np.multiply(np.ones_like(pict, 'float32'), 65)
            fg = np.copy(pict)
        else:
            resultant = cv2.absdiff(pict, self.bg)
            fg = np.where(np.greater(resultant, self.threshold), 255, 0)
            new_bg = np.where(
                np.equal(fg, 0)
                , np.add(((1 - self.alpha) * self.bg), (self.alpha * pict))
                , np.add(((1 - self.beta) * self.bg), (self.beta * pict))
            )
            self.threshold = self.threshold_update(fg, pict)
            print self.threshold
            self.bg = np.uint8(new_bg)
        return cv2.convertScaleAbs(fg)

    def threshold_update(self, fg, pict):
        new_threshold = np.where(
            np.equal(fg, 0),
            np.add(
                np.multiply((1 - self.alpha), self.threshold),
                np.multiply(self.alpha, self.gamma * cv2.absdiff(pict, self.bg))
            ),
            self.threshold
        )
        return new_threshold

    def run(self):
        BackgroundSubtraction.run(self)


# class Median Recursive
class MedianRecursive(BackgroundSubtraction):
    def __init__(self, filename):
        print "initializing Median Recursive..."
        BackgroundSubtraction.__init__(self, filename, True)
        return

    def apply(self, cur_frame):
        new_bg = np.where(np.less_equal(self.bg, cur_frame), self.bg + 1, self.bg - 1)
        return cv2.convertScaleAbs(new_bg)

    def run(self):
        BackgroundSubtraction.run(self)


# class OnlineKMeans
class OnlineKMeans(BackgroundSubtraction):
    def __init__(self, filename, alpha):
        print "initializing Online K Means..."
        BackgroundSubtraction.__init__(self, filename, False)
        self.alpha = alpha
        self.K = 0
        self.centroids = None
        self.w = None
        _, frame = self.vid_src.read()
        gray_pict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.init_clusters(3, gray_pict)
        return

    def init_clusters(self, k, pict):
        self.K = k
        self.centroids = [np.multiply(np.ones_like(pict, 'float64'), ((256. / k) * i)) for i in range(k)]
        self.w = [np.multiply(np.ones_like(pict, 'float64'), (1. / k))] * k
        return

    def apply(self, pict):

        # get min diff & centroid assigned
        min_diff = np.multiply(np.ones_like(pict, 'float64'), -1)
        assigned = np.zeros_like(pict, 'uint8')
        new_bg = np.multiply(np.ones_like(pict, 'uint8'), 255)

        for i in range(self.K):
            # get diff
            cur_diff = np.multiply(np.ones_like(pict, 'float64'), ((pict - self.centroids[i]) ** 2))
            assigned = np.where(np.logical_or(np.equal(min_diff, -1), np.less(cur_diff, min_diff)), i, assigned)
            min_diff = np.where(np.logical_or(np.equal(min_diff, -1), np.less(cur_diff, min_diff)), cur_diff, min_diff)

        # update the centroids and weight
        for i in range(self.K):
            update_centroids = np.multiply(
                np.ones_like(pict, 'float64'),
                (np.add(self.centroids[i], self.alpha * np.subtract(pict, self.centroids[i])))
            )
            self.centroids[i] = np.where(np.equal(assigned, i), update_centroids, self.centroids[i])
            self.w[i] = np.where(np.equal(assigned, i), np.add(np.multiply((1. - self.alpha), self.w[i]), self.alpha),
                                 np.multiply((1. - self.alpha), self.w[i]))
            new_bg = np.where(np.logical_and(np.equal(assigned, i), np.greater(self.w[i], 1. / self.K)), 0, new_bg)

        return new_bg

    def run(self):
        BackgroundSubtraction.run(self)


# class Single Gaussian
# gaussian (mean, variance, weight) is a numpy array
class SingleGaussian(BackgroundSubtraction):
    def __init__(self, filename, alpha, th):
        print "initializing Single Gaussian..."
        BackgroundSubtraction.__init__(self, filename, False)
        self.alpha = alpha
        self.threshold = th
        self.th_array = None
        self.variance = None
        self.mean = None
        return

    def apply(self, pict):
        if self.mean is None:
            self.bg = np.uint8(pict)
            self.mean = np.uint8(pict)
            self.th_array = np.multiply(np.ones_like(pict, 'float64'), self.threshold)
            self.variance = np.multiply(np.ones_like(pict, 'float64'), 20)

        pdf = np.multiply(
            (1. / (np.sqrt(self.variance * 2 * np.pi))),
            np.exp((-((pict - self.mean) ** 2)) / (2 * self.variance))
        )
        new_pict = np.zeros_like(pict)
        new_pict = np.where(np.less(pdf, self.th_array), 255, new_pict)
        self.bg = np.copy(self.mean)
        self.mean = np.add(((1 - self.alpha) * self.mean), (self.alpha * pict))
        # self.variance = np.add(((1-self.alpha) * self.variance), (self.alpha * ((pict - self.bg) ** 2)))
        return new_pict

    def run(self):
        BackgroundSubtraction.run(self)


# class KDE
# implements KDE with LUT
class KDE(BackgroundSubtraction):
    def __init__(self, filename, alpha, th, kernelnum):
        print "initializing KDE..."
        BackgroundSubtraction.__init__(self, filename, False)
        self.alpha = alpha
        self.threshold = th
        self.th_array = None
        self.kernels = None
        self.init_kernel(kernelnum)
        self.pdf_dict = np.array([self.pdf(i, kernelnum) for i in range(256)])
        return

    def apply(self, pict):
        if self.th_array is None:
            self.th_array = np.multiply(np.ones_like(pict, 'float64'), self.threshold)

        pdf_kernel = map(
            lambda x: np.multiply(
                np.ones_like(pict, 'float32'),
                self.pdf_dict[np.absolute(np.subtract(pict, x))]
            ),
            self.kernels
        )
        pdf_combination = reduce(lambda a, b: np.add(a, b), pdf_kernel)
        fg = np.multiply(
            np.ones_like(pict, 'uint8'),
            np.where(
                np.greater(pdf_combination, self.threshold),
                0,
                255
            ).astype('uint8')
        )
        return fg

    def pdf(self, num, kernel_num):
        variance = 30
        index = -(float(num) ** 2) / (2 * variance ** 2)
        result = np.exp(index) / kernel_num
        return result

    def init_kernel(self, n):
        print "initializing kernels for KDE...."
        cvid = cv2.VideoCapture(self.file)
        length = int(cvid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        print length, "frames"
        _, frame = cvid.read()
        it = 0
        nframe = 0
        self.kernels = []
        # iterating through video, filling kernel with images by 3 frame skip
        while frame is not None and it < n:
            if nframe % 3 == 0:
                gray_pict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.kernels.append(np.copy(gray_pict))
                it += 1
            _, frame = cvid.read()
            nframe += 1
        print "done"
        return

    def run(self):
        BackgroundSubtraction.run(self)


# unfinished classes

# class Gaussian Mixture
class GaussianMixture(BackgroundSubtraction):
    def __init__(self, filename, alpha, k):
        print "initializing Gaussian Mixture..."
        BackgroundSubtraction.__init__(self, filename, False)
        self.alpha = alpha
        self.models = None
        self.K = k
        self.means = None
        self.variance = None
        self.weights = None
        return

    def apply(self, pict):

        return pict

    def run(self):
        BackgroundSubtraction.run(self)


# class HiddenMarkov
class HiddenMarkov(BackgroundSubtraction):
    def __init__(self, filename):
        print "initializing Hidden Markov..."
        BackgroundSubtraction.__init__(self, filename, False)
        return

    def apply(self, pict):
        pass

    def run(self):
        BackgroundSubtraction.run(self)


# class Running Average Color
class RunningAverageColor(BackgroundSubtractionColor):
    # warping matrix -- for determining neighbor pixels
    WARP_MATRIX = [
        np.float32([[1, 0, -3], [0, 1, -3]]),
        np.float32([[1, 0, 0], [0, 1, -3]]),
        np.float32([[1, 0, 3], [0, 1, -3]]),
        np.float32([[1, 0, 3], [0, 1, 0]]),
        np.float32([[1, 0, 3], [0, 1, 3]]),
        np.float32([[1, 0, 0], [0, 1, 3]]),
        np.float32([[1, 0, -3], [0, 1, 3]]),
        np.float32([[1, 0, -3], [0, 1, 0]])
    ]

    def __init__(self, params):
        """
        initialization of Running Average Algorithm
        :param params: tuple consists of alpha and beta
        """
        BackgroundSubtractionColor.__init__(self)
        self.bg = None
        self.prev_frame = None
        self.prev_prev_frame = None
        self.alpha = params[0]
        self.beta = params[1]

    def apply(self, cur_image, cur_objects):
        """
        apply the algorithm for running average in color
        :param cur_image: numpy array; a color image (RGB)
        :param cur_objects: array consists of object squares
        :return new_objects_box: array consists of new object squares
        :return new_fg: binary image consists of image (black and white)
        """

        cols, rows, depth = cur_image.shape

        if self.bg is None:
            self.bg = np.copy(cur_image)

        if self.prev_frame is None:
            self.prev_frame = np.copy(cur_image)
            self.prev_prev_frame = np.copy(cur_image)

        # get neighbor pixels
        neighbor_pixels = map(
            lambda x: cv2.warpAffine(cur_image, x, (cols, rows)),
            self.WARP_MATRIX
        )

        # update background
        new_bg = np.add(((1 - self.alpha) * self.bg), (self.alpha * cur_image))

        # compare neighbor pixel with current background
        # neighbor_pixels_diff = map(
        #     lambda x: np.absolute(np.subtract(new_bg, x)),
        #     neighbor_pixels
        # )

        # get difference at this pixel
        diff = np.absolute(np.subtract(new_bg, cur_image))
        fg_raw = cv2.inRange(cv2.cvtColor(diff.astype('uint8'), cv2.COLOR_BGR2GRAY), 25, 255)
        raw_boxes, new_fg = PostProcessing.foreground_process(fg_raw)
        new_objects_box = PostProcessing.bounding_box_mask(raw_boxes, new_fg)

        cv2.imshow('Background', new_bg.astype('uint8'))
        self.bg = np.copy(new_bg)

        self.prev_prev_frame = np.copy(self.prev_frame)
        self.prev_frame = np.copy(cur_image)
        return new_objects_box, new_fg

    @staticmethod
    def get_difference(cur_image, prev_image, threshold):
        threshold_array = np.multiply(np.ones_like(cur_image, 'uint8'), threshold)
        diff = np.absolute(np.subtract(cur_image, prev_image))
        result = np.multiply(
            np.ones_like(cur_image, 'uint8'),
            np.where(
                np.less(diff, threshold_array),
                0,
                255
            )
        )
        return result


# class Frame Difference Color
class FrameDifferenceColor(BackgroundSubtractionColor):
    def __init__(self, params):
        """
        initialization of Running Average Algorithm
        :param params: tuple consists of threshold
        """
        self.threshold = params[0]
        self.prev_image = None
        return

    def apply(self, cur_image):
        """
        apply the algorithm for running average in color
        :param cur_image: numpy array; a color image (RGB)
        :return new_objects_box: array consists of new object squares
        :return new_fg: binary image consists of image (black and white)
        """
        cur_image_gray = cv2.cvtColor(cur_image, cv2.COLOR_BGR2GRAY)
        if self.prev_image is not None:
            threshold_array = np.multiply(np.ones_like(cur_image_gray, 'uint8'), self.threshold)
            diff = np.absolute(np.subtract(cur_image_gray, self.prev_frame))
            fg_raw = np.multiply(
                np.ones_like(cur_image_gray, 'uint8'),
                np.where(
                    np.less(diff, threshold_array),
                    0,
                    255
                )
            )
            raw_boxes, new_fg = PostProcessing.foreground_process(fg_raw)
            new_objects_box = PostProcessing.bounding_box_mask(raw_boxes, new_fg)
        else:
            new_fg = np.zeros_like(cur_image_gray)
            new_objects_box = []

        self.prev_image = np.copy(cur_image_gray)
        return new_objects_box, new_fg
