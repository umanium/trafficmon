import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import id.trafficmon.etc.ImageProcessing
import cv2
import numpy as np
__author__ = 'Luqman'


class ObjectBlob(object):
    contour = None
    roi = None
    mask = None
    centroid = None
    mean = None
    move_direction = None
    n_frames_in_map = None
    area = None
    chain_code = None

    def __init__(self, contour, image):
        self.contour = contour
        # print "contour:", contour
        if image is not None:
            self.roi = id.trafficmon.etc.ImageProcessing.get_mask_from_contour_2(contour, image)
            # chain = cv2.findContours(self.roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.chain_code = self.extract_chain_code()
            moments = cv2.moments(self.contour)
            self.centroid = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
            self.mean = cv2.mean(image, mask=self.roi)[0]
            # self.area = cv2.contourArea(cv2.approxPolyDP(self.contour, 0.001, True))
            self.area = cv2.contourArea(self.contour)
        self.move_direction = (0, 0)
        self.n_frames_in_map = 0

    def get_contour(self):
        return self.contour

    def get_centroid(self):
        return self.centroid

    def get_mask(self):
        return self.mask

    def get_mean(self):
        return self.mean

    def get_move_direction(self):
        return self.move_direction

    def get_n_frames_in_map(self):
        return self.n_frames_in_map

    def get_roi(self):
        return self.roi

    def get_area(self):
        return self.area

    def get_chain_code(self):
        return self.chain_code

    def get_distance(self, point):
        x0, y0 = self.centroid
        x1, y1 = point
        distance = np.sqrt((x0 - x1)**2 + (y0-y1)**2)
        return distance

    def get_color_distance(self, mean):
        distance = np.abs(self.get_mean()-mean)
        return distance

    def get_blob_distance(self, blob):
        return self.get_distance(blob.get_centroid())

    def get_blob_color_distance(self, blob):
        # return 5
        return self.get_color_distance(blob.get_mean())

    def merge_blob(self, blob, image):
        blank_image = np.zeros_like(image)

        # draw blobs
        cv2.drawContours(blank_image, [self.get_contour(), blob.get_contour()], -1, (255, 255, 255), -1)

        # draw line between two centroid
        cv2.line(blank_image, self.get_centroid(), blob.get_centroid(), (255, 255, 255), 5)
        contours, hierarchy = cv2.findContours(blank_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_blob = ObjectBlob(contours[0], image)
        return new_blob

    def track(self, next_blob, image):
        new_blob = ObjectBlob(self.contour, image)
        # copy contents of blob
        direction = (next_blob.get_centroid()[0] - self.get_centroid()[0]), \
                    (next_blob.get_centroid()[1] - self.get_centroid()[1])
        new_blob.contour = next_blob.get_contour()
        new_blob.roi = next_blob.get_roi()
        new_blob.centroid = next_blob.get_centroid()
        new_blob.mask = next_blob.get_mask()
        new_blob.mean = next_blob.get_mean()
        new_blob.n_frames_in_map = self.n_frames_in_map + 1
        new_blob.move_direction = direction

        return new_blob

    def draw(self, image, ids, is_temporal):
        image_used = np.copy(image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        width = 2
        color = (0, 255, 0)
        if not is_temporal:
            width = 1
            color = (255, 0, 0)
        cv2.drawContours(image_used, [self.contour], -1, color, width)
        prev_point = (self.centroid[0] - self.move_direction[0], self.centroid[1] - self.move_direction[1])
        cv2.line(image_used, self.get_centroid(), prev_point, color, width)
        if is_temporal:
            cv2.putText(image_used, str(ids), self.centroid, font, 0.6, (255, 255, 255), 2)
        return image_used

    def is_similar(self, other_blob):
        if other_blob is None:
            return False
        else:
            color_th = 30
            area_th = 20
            color_diff = self.get_mean() - other_blob.get_mean()
            area_diff = self.get_area() - other_blob.get_area()

            return (np.absolute(color_diff) < color_th) and (np.absolute(area_diff) < area_th)

    def move_blob(self, image):
        new_contour = id.trafficmon.etc.ImageProcessing.move_contour(self.contour, self.move_direction, image.shape)
        new_blob = ObjectBlob(new_contour, image)
        return new_blob

    def extract_chain_code(self):
        chain_code = []
        prev_point = None
        for cur_point in self.contour:
            if prev_point is None:
                [[x, y]] = cur_point
                chain_code.append([[x, y]])
            else:
                cur_chain = id.trafficmon.etc.ImageProcessing.extract_chain_code(prev_point, cur_point)
                chain_code.append(cur_chain[0])
                chain_code.append(cur_chain[1])
            prev_point = cur_point
        return np.array(chain_code)

    def get_chain_code_frequency(self):
        frequency = dict()
        for i in range(7):
            frequency[i] = 0
        for idx, chain_code_content in self.chain_code:
            if idx > 0:
                frequency[chain_code_content[0]] = chain_code_content[1]
        return frequency