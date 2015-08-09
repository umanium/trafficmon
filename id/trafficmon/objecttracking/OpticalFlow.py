import numpy as np
import cv2

from id.trafficmon.objecttracking import ObjectTrackingAbstract


__author__ = 'Luqman'


class OpticalFlowHS(ObjectTrackingAbstract):
    """
    class OpticalFlowHS
    implementation of classical Horn-Schunck optical flow (Horn, 1981)
    """

    def __init__(self):
        ObjectTrackingAbstract.__init__(self, "OpticalFlowHS")

    def pyramid(self, image, n):
        cur_img = np.copy(image)
        image_pyramid = [cur_img]

        for i in range(n):
            # downsampling current image
            height, width, depth = cur_img.shape
            if (height / 2 < 8) or (width / 2 < 8):
                break
            else:
                next_img = self.image_downsampling(cur_img)
                # append to image list
                image_pyramid.append(next_img)
                cur_img = next_img

        return image_pyramid

    @staticmethod
    def image_downsampling(image):

        # image_slice_1_1 = image[0::2, :]
        # image_slice_1_2 = image[1::2, :]
        # if image_slice_1_1.shape[0] > image_slice_1_2.shape[0]:
        #     image_slice_1_1 = image_slice_1_1[:image_slice_1_2.shape[0]]
        #
        # image_slice_1 = np.add(image_slice_1_1, image_slice_1_2).astype(np.float32)
        #
        # image_slice_2_1 = image_slice_1[:, 0::2]
        # image_slice_2_2 = image_slice_1[:, 1::2]
        # if image_slice_2_1.shape[1] > image_slice_2_2.shape[1]:
        #     image_slice_2_1 = image_slice_2_1[:, :image_slice_2_2.shape[1]]
        #
        # image_slice_2 = np.add(image_slice_2_1, image_slice_2_2).astype(np.float32) / 4.
        #
        # result_image = np.copy(image_slice_2.astype(np.uint8))
        # return result_image
        # not working, using opencv instead, maybe will visit later

        return cv2.pyrDown(image)

    @staticmethod
    def resample_flow(flow, obj_shape):

        # use opencv pyrUp / pyrDown for resampling, maybe will visit later
        if obj_shape[0] > flow.shape[0]:
            result = cv2.pyrUp(flow, dstsize=obj_shape)
        else:
            result = cv2.pyrDown(flow, dstsize=obj_shape)
        return result

    def compute_flow(self, image):
        image_pyramid = self.pyramid(image, 5)
        flow = np.zeros_like(image)

        return flow

    def partial_derivative(self, image, prev_flow):

        # init flow
        pass

    def bilinear_interpolation(self):
        pass
