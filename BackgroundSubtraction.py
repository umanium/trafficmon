import cv2
import numpy as np
import PostProcessing
import time
from abc import ABCMeta, abstractmethod

"""
Author: Luqman A. M.
BackgroundSubtraction.py
Background Subtraction Algorithms Object Detection in Video Processing
Frame Difference, Running Average, Median, Online K-Means, 1-G, KDE
"""


class BackgroundSubtraction(object):
    __metaclass__ = ABCMeta

    def __init__(self, filename, background):
        self.file = filename
        self.vid_src = cv2.VideoCapture(self.file)
        self.is_background = background
        self.bg = None
        self.prev_frame = None

    @abstractmethod
    def apply(self, data):
        pass

    @abstractmethod
    def run(self):
        self.vid_src = cv2.VideoCapture(self.file)
        _, frame = self.vid_src.read()
        gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_pict = PostProcessing.histEqualization(gray_pict_raw)
        self.bg = np.uint8(gray_pict)

        # applying background detection
        while frame is not None:
            _, frame = self.vid_src.read()
            if frame is None:
                break

            self.prev_frame = np.copy(gray_pict)
            gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_pict = PostProcessing.histEqualization(gray_pict_raw)

            if self.is_background:
                new_bg = self.apply(gray_pict)

                raw_rects, fg = PostProcessing.foregroundDetection(gray_pict, new_bg)
                rects = PostProcessing.boundingBoxMask(raw_rects, fg)
                self.bg = new_bg
            else:
                fg_raw = self.apply(gray_pict)

                raw_rects, fg = PostProcessing.foregroundProcess(fg_raw)
                rects = PostProcessing.boundingBoxMask(raw_rects, fg)

            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(gray_pict, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # showing
            if self.is_background:
                cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)
            cv2.imshow('img', gray_pict)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.vid_src.release()
        return

