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
    def __init__(self, filename, N):
        print "initializing Frame Difference..."
        self.filename = filename
        self.N = N
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
                , np.add(((1-self.alpha) * self.bg), (self.alpha * curFrame))
                , np.add(((1-self.beta) * self.bg), (self.beta * curFrame))
            )
        # newBg = np.copy(curFrame)
        return cv2.convertScaleAbs(newBg)

    def run(self):
        cVid = cv2.VideoCapture(self.file)
        if cVid.isOpened():
            print "Finally"
        else:
            print "BOOM"
        _, frame = cVid.read()
        
        if frame != None:
            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # grayPict = PostProcessing.histEqualization(grayPictRaw)
            grayPict = np.copy(grayPictRaw)
            self.bg = np.uint8(grayPict)

        # applying background detection
        while frame != None:
            start = time.clock()
            prevFrame = np.copy(grayPict)

            newBg = self.apply(grayPict, prevFrame)
            
            rawRects, fg = PostProcessing.foregroundDetection(grayPict, newBg)
            rects = PostProcessing.boundingBoxMask(rawRects, fg)

            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(grayPict, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # cv2.drawContours(grayPict, rects, -1, (0,255,0), 1)

            end = time.clock()

            print end - start

            # showing
            cv2.imshow('img', grayPict)
            # cv2.imshow('imgNorm', grayPictNorm)
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)

            self.bg = np.copy(newBg)

            _, frame = cVid.read()
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
        newBg = np.where(np.less_equal(self.bg, curFrame), self.bg+1, self.bg-1)
        return cv2.convertScaleAbs(newBg)

    def run(self):
        cVid = cv2.VideoCapture(self.file)
        _,frame = cVid.read()
        grayPict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(grayPict)

        # applying background detection
        while True:
            start = time.clock()
            _,frame = cVid.read()
            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayPict = PostProcessing.histEqualization(grayPictRaw)
            newBg = self.apply(grayPict)

            rects, fg = PostProcessing.foregroundDetection(grayPict, newBg)
            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(grayPict, (x, y), (x+w, y+h), (0, 255, 0), 1)
            
            end = time.clock()

            print end - start

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
class GaussianMixture():

    class MixtureModel(object):

        class Gaussian(object):
            def __init__(self, val, w):
                self.mean = val
                self.variance = 90
                self.weight = w
                return

            def pdf(self, data):
                return (1 / (np.sqrt(self.variance * 2 * np.pi))) * np.exp((-((data - self.mean) ** 2)) / (2 * self.variance))

            def match(self, data):
                diff = np.absolute(data - self.mean)
                stdev = np.sqrt(self.variance)
                if diff < (stdev * 2.5):
                    return True
                else:
                    return False

            def update(self, data, alpha, M):
                newWeight = ((1 - alpha) * self.weight) + (alpha * M)
                self.weight = newWeight
                rho = alpha * self.pdf(data)
                newMean = ((1 - rho) * self.mean) + (rho * data)
                self.mean = newMean
                newVariance = ((1 - rho) * self.variance) + (rho * ((data - self.mean) ** 2))
                self.variance = variance
                return

        def __init__(self, K):
            self.K = K
            return
            
    def __init__(self, filename, alpha, K):
        print "initializing Gaussian Mixture..."
        self.file = filename
        self.alpha = alpha
        self.bg = None
        self.models = self.MixtureModel(K)
        return

    def apply(self):
        return

    def run(self):
        return