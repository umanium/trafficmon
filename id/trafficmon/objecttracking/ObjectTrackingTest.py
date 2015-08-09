import cv2
import numpy as np

from id.trafficmon.objecttracking import FunctionsTest


__author__ = 'Luqman'

if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        video_src_file = sys.argv[1]
    else:
        video_src_file = 0

    # run video
    vid_src = cv2.VideoCapture(video_src_file)
    _, frame = vid_src.read()

    # applying background detection
    while frame is not None:

        prev_frame = np.copy(frame)
        _, frame = vid_src.read()

        FunctionsTest.image_pyramid_test(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break