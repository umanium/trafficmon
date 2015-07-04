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
        self.initThreshold = threshold
        return

    def apply(self, data, prevData):
        diff = np.absolute(np.subtract(data, prevData))
        newFg = np.zeros_like(data, "uint8")

        newFg = np.where(np.less(diff, self.threshold), newFg, 255)

        return newFg

    def run(self):
        cVid = cv2.VideoCapture(self.file)
        _, frame = cVid.read()

        if frame is not None:
            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayPict = PostProcessing.histEqualization(grayPictRaw)
            # grayPict = np.copy(grayPictRaw)
            self.bg = np.uint8(grayPict)
            prevFrame = np.copy(grayPict)
            self.threshold = np.multiply(np.ones_like(grayPict, "uint8"), self.initThreshold)

        # applying background detection
        while frame is not None:
            start = time.clock()

            # grayPict = PostProcessing.histEqualization(grayPictRaw)

            fgRaw = self.apply(grayPict, prevFrame)

            rawRects, fg = PostProcessing.foregroundProcess(fgRaw)
            rects = PostProcessing.boundingBoxMask(rawRects, fg)

            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(grayPict, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.drawContours(grayPict, rects, -1, (0,255,0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('img', grayPict)
            # cv2.imshow('imgNorm', grayPictNorm)
            cv2.imshow('Foreground', fg)

            prevFrame = np.copy(grayPict)

            _, frame = cVid.read()
            if frame is None:
                break

            grayPict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cVid.release()
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

    def apply(self, curFrame, prevFrame):
        rects, fg = PostProcessing.foregroundDetection(curFrame, self.bg, False)
        newBg = np.zeros_like(curFrame, 'uint8')
        newBg = np.where(
            np.equal(fg, 0)
            , np.add(((1 - self.alpha) * self.bg), (self.alpha * curFrame))
            , np.add(((1 - self.beta) * self.bg), (self.beta * curFrame))
        )
        # newBg = np.copy(curFrame)
        return cv2.convertScaleAbs(newBg)

    def run(self):
        cVid = cv2.VideoCapture(self.file)
        _, frame = cVid.read()

        if frame is not None:
            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # grayPict = PostProcessing.histEqualization(grayPictRaw)
            grayPict = np.copy(grayPictRaw)
            self.bg = np.uint8(grayPict)

        # applying background detection
        while frame is not None:
            start = time.clock()
            prevFrame = np.copy(grayPict)

            newBg = self.apply(grayPict, prevFrame)

            rawRects, fg = PostProcessing.foregroundDetection(grayPict, newBg)
            rects = PostProcessing.boundingBoxMask(rawRects, fg)

            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(grayPict, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv2.drawContours(grayPict, rects, -1, (0,255,0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('img', grayPict)
            # cv2.imshow('imgNorm', grayPictNorm)
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)

            self.bg = np.copy(newBg)

            _, frame = cVid.read()
            if frame is None:
                break

            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayPict = PostProcessing.histEqualization(grayPictRaw)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cVid.release()
        return


# class Median Recursive
class MedianRecursive():
    def __init__(self, filename):
        print "initializing Median Recursive..."
        self.file = filename
        self.bg = None
        return

    def apply(self, curFrame):
        newBg = np.zeros_like(curFrame, 'uint8')
        newBg = np.where(np.less_equal(self.bg, curFrame), self.bg + 1, self.bg - 1)
        return cv2.convertScaleAbs(newBg)

    def run(self):
        cVid = cv2.VideoCapture(self.file)
        _, frame = cVid.read()
        grayPict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(grayPict)

        # applying background detection
        while True:
            start = time.clock()
            _, frame = cVid.read()
            if frame is None:
                break

            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayPict = PostProcessing.histEqualization(grayPictRaw)
            newBg = self.apply(grayPict)

            rawRects, fg = PostProcessing.foregroundDetection(grayPict, newBg)
            rects = PostProcessing.boundingBoxMask(rawRects, fg)

            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(grayPict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('img', grayPict)
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)

            self.bg = np.copy(newBg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cVid.release()
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
                M = np.where(distance > (2.5 * np.sqrt(self.variance)), 0, 1)

                # get new attributes
                newWeight = (self.weight * (1 - alpha)) + (alpha * M)
                self.weight = newWeight
                rho = self.pdf(data) * alpha
                newMean = ((1 - rho) * self.mean) + (rho * data)
                self.mean = newMean
                newVariance = ((1 - rho) * self.variance) + (rho * ((data - self.mean) ** 2))
                self.variance = newVariance
                return

            def printGaussian(self):
                print self.weight

        def __init__(self, K, pict):
            self.K = K
            self.gaussians = np.array([self.Gaussian((i * (256. / K)), (1. / K), 50, pict) for i in range(K)])
            return

        def updateGaussian(self, data):
            for g in self.gaussians:
                g.update(data, 0.03)
                # g.printGaussian()
            return

        def printModel(self):
            for g in self.gaussians:
                g.printGaussian()

    def __init__(self, filename, alpha, K):
        print "initializing Gaussian Mixture..."
        self.file = filename
        self.alpha = alpha
        self.bg = None
        self.models = None
        self.K = K
        return

    def apply(self, pict):
        self.models.updateGaussian(pict)
        return pict

    def run(self):
        cVid = cv2.VideoCapture(self.file)
        _, frame = cVid.read()
        grayPict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(grayPict)

        # build gaussian mixture models
        self.models = self.MixtureModel(3, grayPict)
        # print grayPict.shape, self.models.gaussians[0].mean.shape

        # applying background detection
        while True:
            start = time.clock()
            _, frame = cVid.read()
            if frame is None:
                break

            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayPict = PostProcessing.histEqualization(grayPictRaw)
            newBg = self.apply(grayPict)

            rects, fg = PostProcessing.foregroundDetection(grayPict, newBg)
            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(grayPict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)
            cv2.imshow('img', grayPict)

            self.bg = np.copy(newBg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cVid.release()
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

    def initClusters(self, K, pict):
        self.K = K
        self.centroids = [np.multiply(np.ones_like(pict, 'float64'), ((256. / K) * i)) for i in range(K)]
        self.w = [np.multiply(np.ones_like(pict, 'float64'), (1. / K)) for i in range(K)]
        self.totalN = K
        return

    def apply(self, pict):
        self.totalN += 1

        # get min diff & centroid assigned
        minDiff = np.multiply(np.ones_like(pict, 'float64'), -1)
        assigned = np.zeros_like(pict, 'uint8')
        newBg = np.multiply(np.ones_like(pict, 'uint8'), 255)

        for i in range(self.K):
            # get diff
            curDiff = np.multiply(np.ones_like(pict, 'float64'), ((pict - self.centroids[i]) ** 2))
            assigned = np.where(np.logical_or(np.equal(minDiff, -1), np.less(curDiff, minDiff)), i, assigned)
            minDiff = np.where(np.logical_or(np.equal(minDiff, -1), np.less(curDiff, minDiff)), curDiff, minDiff)

        # update the centroids and weight
        for i in range(self.K):
            updateCentroids = np.multiply(np.ones_like(pict, 'float64'), (
                np.add(self.centroids[i], self.alpha * np.subtract(pict, self.centroids[i]))))
            self.centroids[i] = np.where(np.equal(assigned, i), updateCentroids, self.centroids[i])
            self.w[i] = np.where(np.equal(assigned, i), np.add(np.multiply((1. - self.alpha), self.w[i]), self.alpha),
                                 np.multiply((1. - self.alpha), self.w[i]))
            newBg = np.where(np.logical_and(np.equal(assigned, i), np.greater(self.w[i], 1. / self.K)), 0, newBg)

        return newBg

    def run(self):
        cVid = cv2.VideoCapture(self.file)
        _, frame = cVid.read()
        grayPict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(grayPict)

        self.initClusters(3, grayPict)

        # applying background detection
        while True:
            start = time.clock()
            _, frame = cVid.read()
            if frame is None:
                break

            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayPict = PostProcessing.histEqualization(grayPictRaw)
            fgRaw = self.apply(grayPict)

            rawRects, fg = PostProcessing.foregroundProcess(fgRaw)
            rects = PostProcessing.boundingBoxMask(rawRects, fg)

            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(grayPict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            end = time.clock()

            # print end - start

            # showing
            # cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)
            cv2.imshow('img', grayPict)

            # self.bg = np.copy(newBg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cVid.release()
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
        self.thArray = None
        self.variance = None
        self.mean = None
        return

    def apply(self, pict):
        pdf = np.multiply((1. / (np.sqrt(self.variance * 2 * np.pi))),
                          np.exp((-((pict - self.mean) ** 2)) / (2 * self.variance)))
        newPict = np.zeros_like(pict)
        newPict = np.where(np.less(pdf, self.thArray), 255, newPict)
        self.bg = np.copy(self.mean)
        self.mean = np.add(((1 - self.alpha) * self.mean), (self.alpha * pict))
        # self.variance = np.add(((1-self.alpha) * self.variance), (self.alpha * ((pict - self.bg) ** 2)))
        return newPict

    def run(self):
        cVid = cv2.VideoCapture(self.file)
        _, frame = cVid.read()
        grayPict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(grayPict)
        self.mean = np.uint8(grayPict)
        self.thArray = np.multiply(np.ones_like(grayPict, 'float64'), self.threshold)
        self.variance = np.multiply(np.ones_like(grayPict, 'float64'), 20)

        # applying background detection
        while True:
            start = time.clock()
            _, frame = cVid.read()
            if frame is None:
                break

            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayPict = PostProcessing.histEqualization(grayPictRaw)
            fgRaw = self.apply(grayPict)

            rawRects, fg = PostProcessing.foregroundProcess(fgRaw)
            rects = PostProcessing.boundingBoxMask(rawRects, fg)
            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(grayPict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)
            cv2.imshow('img', grayPict)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cVid.release()
        return


# class KDE
# implements KDE with LUT and fastbreak
class KDE():
    def __init__(self, filename, alpha, th, kernelNum):
        print "initializing KDE..."
        self.file = filename
        self.alpha = alpha
        self.bg = None
        self.threshold = th
        self.thArray = None
        self.N = kernelNum
        self.kernels = None
        self.kernelInit()
        return

    def apply(self, pict):
        return pict

    def kernelInit(self):
        print "initializing kernels for KDE...."
        cVid = cv2.VideoCapture(self.file)
        length = int(cVid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        print length, "frames"
        _, frame = cVid.read()
        grayPict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        zeros = np.zeros_like(grayPict)
        self.kernels = []
        for i in range(self.N):
            self.kernels.append(np.copy(zeros))
        # self.kernels = np.array([zeros for i in range(self.N)])
        print "done"
        return

    def run(self):
        cVid = cv2.VideoCapture(self.file)
        _, frame = cVid.read()
        grayPict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(grayPict)
        self.thArray = np.multiply(np.ones_like(grayPict, 'float64'), self.threshold)

        # applying background detection
        while True:
            start = time.clock()
            _, frame = cVid.read()
            if frame is None:
                break

            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayPict = PostProcessing.histEqualization(grayPictRaw)
            fgRaw = self.apply(grayPict)

            rawRects, fg = PostProcessing.foregroundProcess(fgRaw)
            rects = PostProcessing.boundingBoxMask(rawRects, fg)
            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(grayPict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            end = time.clock()

            # print end - start

            # showing
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)
            cv2.imshow('img', grayPict)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cVid.release()

        return