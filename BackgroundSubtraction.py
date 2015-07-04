import cv2
import numpy as np
import PostProcessing
import time

"""
Author: Luqman A. M.
BackgroundSubtraction.py
Background Subtraction Algorithms Object Detection in Video Processing
Frame Difference, Running Average, Median, GMM, KDE
"""


# class Frame Difference
class FrameDifference():
    def __init__(self, filename, threshold):
        print "initializing Frame Difference..."
        self.file = filename
        self.init_threshold = threshold
        self.bg = None
        self.threshold = None
        return

    def apply(self, data, prevdata):
        diff = np.absolute(np.subtract(data, prevdata))
        new_fg = np.zeros_like(data, "uint8")

        new_fg = np.where(np.less(diff, self.threshold), new_fg, 255)

        return new_fg

    def run(self):
        cvid = cv2.VideoCapture(self.file)
        _, frame = cvid.read()

        if frame is not None:
            gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_pict = PostProcessing.histEqualization(gray_pict_raw)
            # gray_pict = np.copy(gray_pict_raw)
            self.bg = np.uint8(gray_pict)
            prev_frame = np.copy(gray_pict)
            self.threshold = np.multiply(np.ones_like(gray_pict, "uint8"), self.init_threshold)

        # applying background detection
        while frame is not None:
            start = time.clock()

            # gray_pict = PostProcessing.histEqualization(gray_pict_raw)

            fg_raw = self.apply(gray_pict, prev_frame)

            raw_rects, fg = PostProcessing.foregroundProcess(fg_raw)
            rects = PostProcessing.boundingBoxMask(raw_rects, fg)

            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(gray_pict, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.drawContours(gray_pict, rects, -1, (0,255,0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('img', gray_pict)
            # cv2.imshow('imgNorm', gray_pictNorm)
            cv2.imshow('Foreground', fg)

            prev_frame = np.copy(gray_pict)

            _, frame = cvid.read()
            if frame is None:
                break

            gray_pict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cvid.release()
        return


# class Running Average
class RunningAverage():
    def __init__(self, filename, alpha):
        print "initializing Running Average..."
        self.alpha = alpha
        self.beta = 0.02
        self.file = filename
        self.bg = None
        return

    def apply(self, cur_frame, prev_frame):
        rects, fg = PostProcessing.foregroundDetection(cur_frame, self.bg, False)
        new_bg = np.where(
            np.equal(fg, 0)
            , np.add(((1 - self.alpha) * self.bg), (self.alpha * cur_frame))
            , np.add(((1 - self.beta) * self.bg), (self.beta * cur_frame))
        )
        # new_bg = np.copy(cur_frame)
        return cv2.convertScaleAbs(new_bg)

    def run(self):
        cvid = cv2.VideoCapture(self.file)
        _, frame = cvid.read()

        if frame is not None:
            gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # gray_pict = PostProcessing.histEqualization(gray_pict_raw)
            gray_pict = np.copy(gray_pict_raw)
            self.bg = np.uint8(gray_pict)

        # applying background detection
        while frame is not None:
            start = time.clock()
            prev_frame = np.copy(gray_pict)

            new_bg = self.apply(gray_pict, prev_frame)

            raw_rects, fg = PostProcessing.foregroundDetection(gray_pict, new_bg)
            rects = PostProcessing.boundingBoxMask(raw_rects, fg)

            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(gray_pict, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.drawContours(gray_pict, rects, -1, (0,255,0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('img', gray_pict)
            # cv2.imshow('imgNorm', gray_pictNorm)
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)

            self.bg = np.copy(new_bg)

            _, frame = cvid.read()
            if frame is None:
                break

            gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_pict = PostProcessing.histEqualization(gray_pict_raw)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cvid.release()
        return


# class Median Recursive
class MedianRecursive():
    def __init__(self, filename):
        print "initializing Median Recursive..."
        self.file = filename
        self.bg = None
        return

    def apply(self, cur_frame):
        new_bg = np.zeros_like(cur_frame, 'uint8')
        new_bg = np.where(np.less_equal(self.bg, cur_frame), self.bg + 1, self.bg - 1)
        return cv2.convertScaleAbs(new_bg)

    def run(self):
        cvid = cv2.VideoCapture(self.file)
        _, frame = cvid.read()
        gray_pict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(gray_pict)

        # applying background detection
        while True:
            start = time.clock()
            _, frame = cvid.read()
            if frame is None:
                break

            gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_pict = PostProcessing.histEqualization(gray_pict_raw)
            new_bg = self.apply(gray_pict)

            raw_rects, fg = PostProcessing.foregroundDetection(gray_pict, new_bg)
            rects = PostProcessing.boundingBoxMask(raw_rects, fg)

            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(gray_pict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('img', gray_pict)
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)

            self.bg = np.copy(new_bg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cvid.release()
        return


# class Gaussian Mixture
# unfinished, change the implementation, with gaussian (mean, variance, weight) is a numpy array
class GaussianMixture():
    class MixtureModel(object):
        class Gaussian(object):
            def __init__(self, val, w, var, pict):
                self.mean = np.multiply(np.ones_like(pict, 'float64'), val)
                self.variance = np.multiply(np.ones_like(pict, 'float64'), var)
                self.weight = np.multiply(np.ones_like(pict, 'float64'), w)
                return

            def pdf(self, data):
                return np.multiply((1. / (np.sqrt(self.variance * 2 * np.pi))),
                                   np.exp((-((data - self.mean) ** 2)) / (2 * self.variance)))

            def match(self, data):
                diff = np.absolute(data - self.mean)
                stdev = np.sqrt(self.variance)
                if diff < (stdev * 2.5):
                    return True
                else:
                    return False

            def update(self, data, alpha):
                # do calculations
                distance = np.subtract(data, self.mean)
                m = np.where(distance > (2.5 * np.sqrt(self.variance)), 0, 1)

                # get new attributes
                new_weight = (self.weight * (1 - alpha)) + (alpha * m)
                self.weight = new_weight
                rho = self.pdf(data) * alpha
                new_mean = ((1 - rho) * self.mean) + (rho * data)
                self.mean = new_mean
                new_variance = ((1 - rho) * self.variance) + (rho * ((data - self.mean) ** 2))
                self.variance = new_variance
                return

            def print_gaussian(self):
                print self.weight

        def __init__(self, k, pict):
            self.K = k
            self.gaussians = np.array([self.Gaussian((i * (256. / k)), (1. / k), 50, pict) for i in range(k)])
            return

        def update_gaussian(self, data):
            for g in self.gaussians:
                g.update(data, 0.03)
                # g.printGaussian()
            return

        def print_model(self):
            for g in self.gaussians:
                g.print_gaussian()

    def __init__(self, filename, alpha, K):
        print "initializing Gaussian Mixture..."
        self.file = filename
        self.alpha = alpha
        self.bg = None
        self.models = None
        self.K = K
        return

    def apply(self, pict):
        self.models.update_gaussian(pict)
        return pict

    def run(self):
        cvid = cv2.VideoCapture(self.file)
        _, frame = cvid.read()
        gray_pict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(gray_pict)

        # build gaussian mixture models
        self.models = self.MixtureModel(3, gray_pict)
        # print gray_pict.shape, self.models.gaussians[0].mean.shape

        # applying background detection
        while True:
            start = time.clock()
            _, frame = cvid.read()
            if frame is None:
                break

            gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_pict = PostProcessing.histEqualization(gray_pict_raw)
            new_bg = self.apply(gray_pict)

            rects, fg = PostProcessing.foregroundDetection(gray_pict, new_bg)
            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(gray_pict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)
            cv2.imshow('img', gray_pict)

            self.bg = np.copy(new_bg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cvid.release()
        return


# class HiddenMarkov
class HiddenMarkov():
    def __init__(self, filename):
        print "initializing Hidden Markov..."
        self.file = filename
        return


# class OnlineKMeans
class OnlineKMeans():
    def __init__(self, filename, alpha):
        print "initializing Online K Means..."
        self.file = filename
        self.alpha = alpha
        self.bg = None
        self.K = 0
        self.centroids = None
        self.w = None
        self.totalN = self.K
        return

    def init_clusters(self, K, pict):
        self.K = K
        self.centroids = [np.multiply(np.ones_like(pict, 'float64'), ((256. / K) * i)) for i in range(K)]
        self.w = [np.multiply(np.ones_like(pict, 'float64'), (1. / K)) for i in range(K)]
        self.totalN = K
        return

    def apply(self, pict):
        self.totalN += 1

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
            update_centroids = np.multiply(np.ones_like(pict, 'float64'), (
                np.add(self.centroids[i], self.alpha * np.subtract(pict, self.centroids[i]))))
            self.centroids[i] = np.where(np.equal(assigned, i), update_centroids, self.centroids[i])
            self.w[i] = np.where(np.equal(assigned, i), np.add(np.multiply((1. - self.alpha), self.w[i]), self.alpha),
                                 np.multiply((1. - self.alpha), self.w[i]))
            new_bg = np.where(np.logical_and(np.equal(assigned, i), np.greater(self.w[i], 1. / self.K)), 0, new_bg)

        return new_bg

    def run(self):
        cvid = cv2.VideoCapture(self.file)
        _, frame = cvid.read()
        gray_pict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(gray_pict)

        self.init_clusters(3, gray_pict)

        # applying background detection
        while True:
            start = time.clock()
            _, frame = cvid.read()
            if frame is None:
                break

            gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_pict = PostProcessing.histEqualization(gray_pict_raw)
            fg_raw = self.apply(gray_pict)

            raw_rects, fg = PostProcessing.foregroundProcess(fg_raw)
            rects = PostProcessing.boundingBoxMask(raw_rects, fg)

            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(gray_pict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            end = time.clock()

            # print end - start

            # showing
            # cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)
            cv2.imshow('img', gray_pict)

            # self.bg = np.copy(new_bg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cvid.release()
        return


# class Single Gaussian
# gaussian (mean, variance, weight) is a numpy array
class SingleGaussian():
    def __init__(self, filename, alpha, th):
        print "initializing Single Gaussian..."
        self.file = filename
        self.alpha = alpha
        self.bg = None
        self.threshold = th
        self.th_array = None
        self.variance = None
        self.mean = None
        return

    def apply(self, pict):
        pdf = np.multiply((1. / (np.sqrt(self.variance * 2 * np.pi))),
                          np.exp((-((pict - self.mean) ** 2)) / (2 * self.variance)))
        new_pict = np.zeros_like(pict)
        new_pict = np.where(np.less(pdf, self.th_array), 255, new_pict)
        self.bg = np.copy(self.mean)
        self.mean = np.add(((1 - self.alpha) * self.mean), (self.alpha * pict))
        # self.variance = np.add(((1-self.alpha) * self.variance), (self.alpha * ((pict - self.bg) ** 2)))
        return new_pict

    def run(self):
        cvid = cv2.VideoCapture(self.file)
        _, frame = cvid.read()
        gray_pict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(gray_pict)
        self.mean = np.uint8(gray_pict)
        self.th_array = np.multiply(np.ones_like(gray_pict, 'float64'), self.threshold)
        self.variance = np.multiply(np.ones_like(gray_pict, 'float64'), 20)

        # applying background detection
        while True:
            start = time.clock()
            _, frame = cvid.read()
            if frame is None:
                break

            gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_pict = PostProcessing.histEqualization(gray_pict_raw)
            fg_raw = self.apply(gray_pict)

            raw_rects, fg = PostProcessing.foregroundProcess(fg_raw)
            rects = PostProcessing.boundingBoxMask(raw_rects, fg)
            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(gray_pict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)
            cv2.imshow('img', gray_pict)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cvid.release()
        return


# class KDE
# implements KDE with LUT and fastbreak
class KDE():
    def __init__(self, filename, alpha, th, kernelnum):
        print "initializing KDE..."
        self.file = filename
        self.alpha = alpha
        self.bg = None
        self.threshold = th
        self.th_array = None
        self.N = kernelnum
        self.kernels = None
        self.init_kernel()
        return

    def apply(self, pict):
        return pict

    def init_kernel(self):
        print "initializing kernels for KDE...."
        cvid = cv2.VideoCapture(self.file)
        length = int(cvid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        print length, "frames"
        _, frame = cvid.read()
        gray_pict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zeros = np.zeros_like(gray_pict)
        self.kernels = []
        for i in range(self.N):
            self.kernels.append(np.copy(zeros))
        # self.kernels = np.array([zeros for i in range(self.N)])
        print "done"
        return

    def run(self):
        cvid = cv2.VideoCapture(self.file)
        _, frame = cvid.read()
        gray_pict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(gray_pict)
        self.th_array = np.multiply(np.ones_like(gray_pict, 'float64'), self.threshold)

        # applying background detection
        while True:
            start = time.clock()
            _, frame = cvid.read()
            if frame is None:
                break

            gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_pict = PostProcessing.histEqualization(gray_pict_raw)
            fg_raw = self.apply(gray_pict)

            raw_rects, fg = PostProcessing.foregroundProcess(fg_raw)
            rects = PostProcessing.boundingBoxMask(raw_rects, fg)
            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(gray_pict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)
            cv2.imshow('img', gray_pict)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cvid.release()

        return