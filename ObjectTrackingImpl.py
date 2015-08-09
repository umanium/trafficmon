import cv2
import numpy as np
import math
from ObjectTracking import ObjectTracking
import PostProcessing

__author__ = 'Luqman'

"""
Author: Luqman A. M.
ObjectTracking.py
Object Tracking Algorithms in Video Processing (Implementation)
Optical Flow, Lucas-Kanade-Tomassi, MeanShift, CAMShift
"""


class MeanShift(ObjectTracking):
    def __init__(self):
        ObjectTracking.__init__(self)

    def run(self, rect, cur_frame, next_frame):
        x, y, w, h = rect
        cur_roi = PostProcessing.get_roi_from_images(rect, cur_frame)
        center_of_window = (x + (w / 2), y + (h / 2))

        # compute centroid of current frame
        cur_moment = cv2.moments(cur_roi)
        cx = x + int(cur_moment['m10'] / cur_moment['m00'])
        cy = y + int(cur_moment['m01'] / cur_moment['m00'])
        cur_frame_centroid = (cx, cy)

        # compute centroid of next frame with current windows
        cur_roi_next = PostProcessing.get_roi_from_images(rect, next_frame)
        cur_moment_next = cv2.moments(cur_roi_next)
        next_cx = x + int(cur_moment_next['m10'] / cur_moment_next['m00'])
        next_cy = y + int(cur_moment_next['m01'] / cur_moment_next['m00'])
        next_frame_centroid = (next_cx, next_cy)

        # calculate distance between current frame centroid and next frame centroid
        x0, y0 = cur_frame_centroid
        x1, y1 = next_frame_centroid
        xwin, ywin = center_of_window
        new_center_of_window = ((xwin + (x1 - x0)), (ywin + (y1 - y0)))
        new_rect = (new_center_of_window[0] - (w / 2), new_center_of_window[1] - (h / 2), w, h)
        print new_rect

        pass


class MeanShift2(ObjectTracking):
    def __init__(self):
        ObjectTracking.__init__(self)
        self.list_of_objects = []

    def run(self, cur_frame, next_frame,):
        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        new_list_of_objects = []

        # normal camshift with only one list of objects
        for obj_tuple in self.list_of_objects:
            hsv_roi = None
            if len(obj_tuple) == 4:
                obj, hsv_roi, n_in_frame, n_not_moving = obj_tuple
            if (hsv_roi is not None) and (obj[2] > 0 or obj[3] > 0):
                mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

                # track in next frame
                # backprojection
                hsv = cv2.cvtColor(next_frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                # apply meanshift to get the new location
                # ret, obj_new = cv2.meanShift(dst, obj, term_crit)
                obj_new = self.apply_meanshift(obj, cur_frame, next_frame)
                n_in_frame += 1
                if PostProcessing.distance_two_squares(obj, obj_new) < 2:
                    n_not_moving += 1
                else:
                    n_not_moving = 0

                x, y, w, h = obj_new
                if n_not_moving < 20:
                    new_list_of_objects.append((obj_new, hsv_roi, n_in_frame, n_not_moving))

                # draw
                cv2.rectangle(next_frame, (x, y), (x + w, y + h), 255, 2)
        self.list_of_objects = new_list_of_objects
        pass

    def add_object(self, obj, frame):
        # set up ROI
        roi = PostProcessing.get_roi_from_images(obj, frame)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        n_in_frame = 0
        n_not_moving = 0
        self.list_of_objects.append((obj, hsv_roi, n_in_frame, n_not_moving))

    def is_object_empty(self):
        return not self.list_of_objects

    def objects(self):
        return [obj[0] for obj in self.list_of_objects]

    def apply_meanshift(self, obj, cur_frame, next_frame):
        new_obj = None
        if next_frame is not None and cur_frame is not None:
            cur_frame_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
            next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = obj
            if w > 0 and h > 0:
                cur_roi = PostProcessing.get_roi_from_images(obj, cur_frame_gray)
                # center_of_window = (x + (w / 2), y + (h / 2))

                # compute centroid of current frame
                cur_moment = cv2.moments(cur_roi)
                cx0 = x + int(cur_moment['m10'] / cur_moment['m00'])
                cy0 = y + int(cur_moment['m01'] / cur_moment['m00'])

                num_of_iteration = 0
                delta = -1
                prev_obj = obj
                while (num_of_iteration < 15) and (delta > 1 or delta == -1):

                    x1, y1, w1, h1 = prev_obj
                    next_frame_roi = PostProcessing.get_roi_from_images(prev_obj, next_frame_gray)
                    next_w, next_h = next_frame_roi.shape
                    if next_w > 0 and next_h > 0:

                        # get moment
                        next_frame_moment = cv2.moments(next_frame_roi)
                        cx1 = x1 + int(next_frame_moment['m10'] / next_frame_moment['m00'])
                        cy1 = y1 + int(next_frame_moment['m01'] / next_frame_moment['m00'])

                        # compare with previous moment
                        deltacx = cx1 - cx0
                        deltacy = cy1 - cy0
                        new_obj = x1+deltacx, y1+deltacy, w1, h1

                        # initialization for next iteration
                        cx0, cy0 = cx1, cy1
                        prev_obj = new_obj
                        delta = math.sqrt((deltacx)**2 + (deltacy)**2)
                    num_of_iteration += 1

        return new_obj


class CamShift(ObjectTracking):
    def __init__(self):
        ObjectTracking.__init__(self)
        self.list_of_objects = []

    def run(self, cur_frame, next_frame,):
        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        new_list_of_objects = []

        for obj_tuple in self.list_of_objects:
            hsv_roi = None
            if len(obj_tuple) == 4:
                obj, hsv_roi, n_in_frame, n_not_moving = obj_tuple
            if (hsv_roi is not None) and (obj[2] > 0 or obj[3] > 0):
                mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
                roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
                cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

                # track in next frame
                # backprojection
                hsv = cv2.cvtColor(next_frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

                # apply meanshift to get the new location
                ret, obj_new = cv2.meanShift(dst, obj, term_crit)
                n_in_frame += 1
                if PostProcessing.distance_two_squares(obj, obj_new) < 1:
                    n_not_moving += 1
                else:
                    n_not_moving = 0

                x, y, w, h = obj_new
                if n_not_moving < 20:
                    new_list_of_objects.append((obj_new, hsv_roi, n_in_frame, n_not_moving))

                # draw
                cv2.rectangle(next_frame, (x, y), (x + w, y + h), 255, 2)
        self.list_of_objects = new_list_of_objects
        pass

    def add_object(self, obj, frame):
        # set up ROI
        roi = PostProcessing.get_roi_from_images(obj, frame)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        n_in_frame = 0
        n_not_moving = 0
        self.list_of_objects.append((obj, hsv_roi, n_in_frame, n_not_moving))

    def is_object_empty(self):
        return not self.list_of_objects

    def objects(self):
        return [obj[0] for obj in self.list_of_objects]


class OpticalFlow(ObjectTracking):
    # warping matrix -- for determining neighbor pixels
    WARP_MATRIX = [
        np.float32([[1, 0, -1], [0, 1, -1]]),
        np.float32([[1, 0, 0], [0, 1, -1]]),
        np.float32([[1, 0, 1], [0, 1, -1]]),
        np.float32([[1, 0, 1], [0, 1, 0]]),
        np.float32([[1, 0, 1], [0, 1, 1]]),
        np.float32([[1, 0, 0], [0, 1, 1]]),
        np.float32([[1, 0, -1], [0, 1, 1]]),
        np.float32([[1, 0, -1], [0, 1, 0]])
    ]

    def __init__(self):
        ObjectTracking.__init__(self)
        self.prev_frame = None

    def get_average(self):
        average = None
        if self.prev_frame is not None:
            pass
        return average

    def run(self):
        pass