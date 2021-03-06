import cv2
import numpy as np
import PostProcessing
from abc import ABCMeta, abstractmethod

"""
Author: Luqman A. M.
BackgroundSubtraction.py
Background Subtraction Algorithms Object Detection in Video Processing (Abstract Class)
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
    def apply(self, pict):
        pass

    @abstractmethod
    def run(self):
        self.vid_src = cv2.VideoCapture(self.file)
        _, frame = self.vid_src.read()
        gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_pict = PostProcessing.hist_equalization(gray_pict_raw)
        self.bg = np.copy(gray_pict)

        # applying background detection
        while frame is not None:
            _, frame = self.vid_src.read()
            if frame is None:
                break

            self.prev_frame = np.copy(gray_pict)
            gray_pict_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_pict = PostProcessing.hist_equalization(gray_pict_raw)

            if self.is_background:
                new_bg = self.apply(gray_pict)

                raw_rects, fg = PostProcessing.foreground_detection(gray_pict, new_bg)
                rects = PostProcessing.bounding_box_mask(raw_rects, fg)
                self.bg = new_bg
            else:
                fg_raw = self.apply(gray_pict)
                raw_rects, fg = PostProcessing.foreground_process(fg_raw)
                rects = PostProcessing.bounding_box_mask(raw_rects, fg)

            roi_imgs = []
            moments = []
            # print rects
            for box in rects:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                roi = PostProcessing.get_roi_from_images(box, gray_pict)
                # roi_imgs.append(roi)
                cur_moment = cv2.moments(roi)
                # moments.append(cur_moment)
                cx = x + int(cur_moment['m10'] / cur_moment['m00'])
                cy = y + int(cur_moment['m01'] / cur_moment['m00'])
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

            # showing
            cv2.imshow('Background', self.bg)
            cv2.imshow('Foreground', fg)
            cv2.imshow('img', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        self.vid_src.release()
        return


class BackgroundSubtractionColor(object):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, cur_image):
        pass