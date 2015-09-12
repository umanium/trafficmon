import os

import cv2
import numpy as np
import time

from backgroundsubtraction.KMeans import KMeans
from objectblob.ObjectBlobDetection import ObjectBlobDetection
from pixelcleaning.MorphologicalCleaning import MorphologicalCleaning


__author__ = 'Luqman'


def morphological(image):
    cleaning_model = MorphologicalCleaning()
    return cleaning_model


def test(algorithm, vid_src, file_name):
    _, frame = vid_src.read()
    used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    model = KMeans(used_frame, 3)
    cleaning_model = algorithm(used_frame)
    blob_detection = ObjectBlobDetection(used_frame)

    n_frame = 0
    image_resolution = (0, 0)

    min_fps = -1
    max_fps = -1
    mean_fps = -1

    real_fps = vid_src.get(cv2.cv.CV_CAP_PROP_FPS)
    # vid_src.get(cv2.CV_CAP_PROP_FPS)

    if not os.path.exists("saved_images/"+file_name):
        os.makedirs("saved_images/"+file_name)
        os.makedirs("saved_images/"+file_name+"/normal")
        os.makedirs("saved_images/"+file_name+"/fg")
        os.makedirs("saved_images/"+file_name+"/grayscale")
        os.makedirs("saved_images/"+file_name+"/clean")
        os.makedirs("saved_images/"+file_name+"/contour")

    # applying background detection
    while frame is not None:
        time_start = time.time()
        n_frame += 1

        # for explanational purpose
        # ambil gambar
        # if n_frame % 30 == 0:
        #     cv2.imwrite("saved_images/"+file_name+"/normal/"+repr(n_frame)+".jpg", frame)

        used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        y, x = used_frame.shape
        image_resolution = x, y

        fg = model.apply(used_frame)

        # for explanational purpose
        # ambil gambar
        # if n_frame % 30 == 0:
        #     cv2.imwrite("saved_images/"+file_name+"/fg/"+repr(n_frame)+".jpg", fg)
        #     cv2.imwrite("saved_images/"+file_name+"/grayscale/"+repr(n_frame)+".jpg", used_frame)

        fg_use = np.copy(fg)
        fg_clean = cleaning_model.apply(fg)
        fg_clean_use = np.copy(fg_clean)

        # for explanational purpose
        # ambil gambar
        # if n_frame % 30 == 0:
        #     cv2.imwrite("saved_images/"+file_name+"/clean/"+repr(n_frame)+".jpg", fg_clean)

        # contours
        blob_detection.get_contours(fg_clean_use, used_frame)
        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        frame_with_contours = blob_detection.draw_blobs(frame)
        # print len(contours)

        # for explanational purpose
        # ambil gambar
        # if n_frame % 30 == 0:
        #     cv2.imwrite("saved_images/"+file_name+"/contour/"+repr(n_frame)+".jpg", frame_with_contours)

        time_end = time.time()

        cv2.imshow('img', frame_with_contours)
        cv2.imshow('fg', fg)
        cv2.imshow('fg_clean', fg_clean)

        # prev_frame = np.copy(frame)
        _, frame = vid_src.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time_process = time_end - time_start
        cur_fps = 0
        if time_process > 0:
            cur_fps = 1. / time_process

        # set max / min / mean fps
        if (cur_fps > max_fps) or (max_fps == -1):
            max_fps = cur_fps
        if (cur_fps < min_fps) or (min_fps == -1):
            min_fps = cur_fps
        if mean_fps == -1:
            mean_fps = cur_fps
        else:
            mean_fps = (0.98 * mean_fps) + (0.02 * cur_fps)

    print "--- run statistics ---"
    print "image resolution: ", image_resolution
    print "total frame: ", n_frame
    print "min FPS: ", min_fps
    print "max FPS: ", max_fps
    print "average FPS: ", mean_fps
    print "Video FPS: ", real_fps


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        video_src_file = sys.argv[1]
        if len(sys.argv) >= 3:
            exp_file_name = sys.argv[2]
        else:
            exp_file_name = "default"
    else:
        video_src_file = 0
        exp_file_name = "default"

    # run video
    vid = cv2.VideoCapture(video_src_file)
    test(morphological, vid, exp_file_name)
