import cv2
import numpy as np
from SingleGaussian import SingleGaussian
from KernelDensityEstimation import KernelDensityEstimation
from KMeans import KMeans
from KalmanFilter import KalmanFilter

__author__ = 'Luqman'


def single_gaussian_test(vid_src):
    _, frame = vid_src.read()
    used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    model = SingleGaussian(used_frame)

    # applying background detection
    while frame is not None:

        used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fg = model.apply(used_frame)

        cv2.imshow('img', used_frame)
        cv2.imshow('fg', fg)

        prev_frame = np.copy(frame)
        _, frame = vid_src.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def kde_test(vid_src):
    _, frame = vid_src.read()
    used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    model = KernelDensityEstimation(used_frame, 50)

    # applying background detection
    while frame is not None:

        used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fg = model.apply(used_frame)

        cv2.imshow('img', used_frame)
        cv2.imshow('fg', fg)

        prev_frame = np.copy(frame)
        _, frame = vid_src.read()

        # print model.get_background_model().get_density_range(used_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pass


def kmeans_test(vid_src):
    _, frame = vid_src.read()
    used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    model = KMeans(used_frame, 3)

    # applying background detection
    while frame is not None:

        used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fg = model.apply(used_frame)

        cv2.imshow('img', used_frame)
        cv2.imshow('fg', fg)

        prev_frame = np.copy(frame)
        _, frame = vid_src.read()

        # print model.get_background_model().get_density_range(used_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pass


def kalman_test(vid_src):
    _, frame = vid_src.read()
    used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    model = KalmanFilter(used_frame, 60)

    # applying background detection
    while frame is not None:

        used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fg = model.apply(used_frame)

        cv2.imshow('img', used_frame)
        cv2.imshow('fg', fg)

        prev_frame = np.copy(frame)
        _, frame = vid_src.read()

        # print model.get_background_model().get_density_range(used_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pass

if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        video_src_file = sys.argv[1]
    else:
        video_src_file = 0

    # run video
    vid = cv2.VideoCapture(video_src_file)
    kalman_test(vid)
