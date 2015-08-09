import cv2
import numpy as np
from SingleGaussian import SingleGaussian

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