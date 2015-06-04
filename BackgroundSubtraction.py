import cv2
import numpy as np
import PostProcessing

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
        self.file = filename
        self.bg = None
        return

    def apply(self, curFrame):
        newBg = np.zeros_like(curFrame, 'uint8')
        newBg = np.add(((1-self.alpha) * self.bg), (self.alpha * curFrame))
        # newBg = np.copy(curFrame)
        return cv2.convertScaleAbs(newBg)

    def run(self):
        cVid = cv2.VideoCapture(self.file)
        _, frame = cVid.read()
        grayPict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.bg = np.uint8(grayPict)

        # applying background detection
        while True:
            _, frame = cVid.read()
            grayPictRaw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayPict = PostProcessing.histEqualization(grayPictRaw)
            newBg = self.apply(grayPict)
            
            rects, fg = PostProcessing.foregroundDetection(grayPict, newBg)
            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(grayPict, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # cv2.drawContours(grayPict, rects, -1, (0,255,0), 1)

            # showing
            cv2.imshow('img', grayPict)
            # cv2.imshow('imgNorm', grayPictNorm)
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)

            self.bg = np.copy(newBg)

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
            _,frame = cVid.read()
            grayPict = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            newBg = self.apply(grayPict)
            
            # showing
            cv2.imshow('img', grayPict)
            cv2.imshow('Background', self.bg)

            self.bg = np.copy(newBg)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cVid.release()
        return