import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
import cv2
from id.trafficmon.objectblob.ObjectBlob import ObjectBlob
from id.trafficmon.objectblob.ObjectBlobManager import ObjectBlobManager
__author__ = 'Luqman'

'''
object blob detection
detecting blob from foreground mask generated from background subtraction
'''


class ObjectBlobDetection(object):

    prev_mask = None
    prev_contours = None
    blob_manager = None
    spatial_blob_manager = None

    def __init__(self, image_mask):
        self.prev_mask = np.copy(image_mask)

    def get_contours(self, image_mask, reference_image):
        contours, hierarchy = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # eliminate all too-small contours
        contours_intermediate = []
        for cnt in contours:
            contour_area = cv2.contourArea(cnt)
            if contour_area > 5:
                hull = cv2.convexHull(cnt, returnPoints=True)
                # intersect = self.get_prev_contour(hull, image_mask)
                contours_intermediate.append(hull)

        # new blob manager generation
        raw_blob_manager = ObjectBlobManager(contours_intermediate, reference_image)

        # spatial evaluation
        merge_list, removed_list = raw_blob_manager.spatial_evaluation()
        spatial_blob_manager = raw_blob_manager.remove_and_merge(removed_list, merge_list)

        # tracking: membandingkan antara contour yg skrg dengan yg sebelumnya
        if self.blob_manager is not None:
            next_blob_manager = spatial_blob_manager.temporal_evaluation(self.blob_manager, reference_image)
        else:
            next_blob_manager = spatial_blob_manager.copy()

        self.spatial_blob_manager = spatial_blob_manager.copy()

        if next_blob_manager is not None:
            self.blob_manager = next_blob_manager.copy()  # next_blob_manager.copy()
        else:
            self.blob_manager = spatial_blob_manager.copy()

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

    def bounding_box_mask(self, contours, img):
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, -1, 1, thickness=-1)

        new_contours = []
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            new_contours.append(cv2.convexHull(cnt, returnPoints=True))
        return new_contours

    def draw_blobs(self, image):
        image_used = np.copy(image)
        if self.blob_manager is not None:
            image_result = self.blob_manager.draw_contours(image_used, True)
        else:
            image_result = np.copy(image_used)

        # if self.spatial_blob_manager is not None:
        #     image_result_2 = self.spatial_blob_manager.draw_contours(image_result, False)
        # else:
        #     image_result_2 = np.copy(image_result)
        return image_result