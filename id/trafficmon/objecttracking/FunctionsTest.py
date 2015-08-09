import cv2

from id.trafficmon.objecttracking.OpticalFlow import OpticalFlowHS


__author__ = 'Luqman'


def image_pyramid_test(image):
    if image is not None:
        pyramid_list = OpticalFlowHS().pyramid(image, 5)
        for idx, img in enumerate(pyramid_list):
            cv2.imshow('img'+str(idx), img)