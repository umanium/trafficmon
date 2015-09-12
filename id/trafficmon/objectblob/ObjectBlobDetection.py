import numpy as np
import cv2
__author__ = 'Luqman'

'''
object blob detection
detecting blob from foreground mask generated from background subtraction
'''


class ObjectBlobDetection(object):

    prev_mask = None
    prev_contours = None

    def __init__(self, image_mask):
        prev_mask = np.copy(image_mask)

    def form_blob(self, image_mask):
        frame_diff = np.absolute(np.subtract(image_mask, self.prev_mask))

    def get_contours(self, image_mask):
        contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # eliminate all too-small contours
        contours_intermediate = []
        for cnt in contours:
            contour_area = cv2.contourArea(cnt)
            if contour_area > 5:
                hull = cv2.convexHull(cnt, returnPoints=True)
                # intersect = self.get_prev_contour(hull, image_mask)
                contours_intermediate.append(hull)

        # tracking: membandingkan antara contour yg skrg dengan yg sebelumnya
        # belom
        self.prev_contours = contours_intermediate

        # for contour in contours:
        #     rects.append(cv2.boundingRect(contour))

        contours_clean = self.bounding_box_mask(contours_intermediate, image_mask)

        return contours_clean

    def get_prev_contour(self, contour, image):
        max_contour_area = 5
        intersect_contour = None
        cur_cont_img = np.zeros_like(image)
        cv2.drawContours(cur_cont_img, np.array([contour]), 0, 1, thickness=-1)

        if self.prev_contours is not None:
            for prev_contour in self.prev_contours:
                prev_cont_img = np.zeros_like(image)
                cv2.drawContours(prev_cont_img, np.array([prev_contour]), 0, 1, thickness=-1)
                intersection = np.logical_and(cur_cont_img, prev_cont_img)
                intersection_area = np.count_nonzero(intersection)
                if intersection_area > max_contour_area:
                    max_contour_area = intersection_area
                    intersect_contour = intersection

        return intersect_contour

    def merge_contours(self, contour1, contour2):
        new_contour = contour1
        return new_contour

    def bounding_box_mask(self, contours, img):
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, -1, 1, thickness=-1)

        newcontours = []
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            newcontours.append(cv2.convexHull(cnt, returnPoints=True))
        return newcontours