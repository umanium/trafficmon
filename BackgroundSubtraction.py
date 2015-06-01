import cv2
import numpy as np

"""
Author: Luqman A. M.
BackgroundSubtraction.py
Background Subtraction Algorithms Object Detection in Video Processing
Frame Difference, Running Average, Median, GMM, KDE
"""

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