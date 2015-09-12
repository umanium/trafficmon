import os.path

from id.trafficmon.objectblob import ObjectBlobDetection


sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import cv2
import numpy as np
from id.trafficmon.backgroundsubtraction.KMeans import KMeans
from id.trafficmon.pixelcleaning.MorphologicalCleaning import MorphologicalCleaning

__author__ = 'Luqman'


def morphological(image):
    cleaning_model = MorphologicalCleaning()
    return cleaning_model


def test(algorithm, vid_src):
    _, frame = vid_src.read()
    used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    model = KMeans(used_frame, 3)
    cleaning_model = algorithm(used_frame)
    blob_detection = ObjectBlobDetection(used_frame)

    # applying background detection
    while frame is not None:

        used_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fg = model.apply(used_frame)
        fg_use = np.copy(fg)
        fg_clean = cleaning_model.apply(fg)
        fg_clean_use = np.copy(fg_clean)

        # contours
        contours = blob_detection.get_contours(fg_use)
        cv2.drawContours(used_frame, contours, -1, (0, 255, 0), 2)
        print len(contours)

        cv2.imshow('img', used_frame)
        cv2.imshow('fg', fg)
        cv2.imshow('fg_clean', fg_clean)

        prev_frame = np.copy(frame)
        _, frame = vid_src.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        video_src_file = sys.argv[1]
    else:
        video_src_file = 0

    # run video
    vid = cv2.VideoCapture(video_src_file)
    test(morphological, vid)