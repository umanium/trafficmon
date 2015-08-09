import cv2
import numpy as np

__author__ = 'Luqman'

if __name__ == '__main__':
    import sys

    if len(sys.argv) >= 2:
        video_src = sys.argv[1]
    else:
        video_src = 0

    vid_src = cv2.VideoCapture(video_src)
    _, frame = vid_src.read()
    object_box = None

    # warping matrix
    MM = [
        np.float32([[1, 0, -3], [0, 1, -3]]),
        np.float32([[1, 0, 0], [0, 1, -3]]),
        np.float32([[1, 0, 3], [0, 1, -3]]),
        np.float32([[1, 0, 3], [0, 1, 0]]),
        np.float32([[1, 0, 3], [0, 1, 3]]),
        np.float32([[1, 0, 0], [0, 1, 3]]),
        np.float32([[1, 0, -3], [0, 1, 3]]),
        np.float32([[1, 0, -3], [0, 1, 0]])
    ]

    rows, cols, depth = frame.shape

    # setup initial location of window
    r, h, c, w = 250, 100, 200, 125  # simply hardcoded the values
    track_window = (c, r, w, h)

    # set up the ROI for tracking
    roi = frame[r:r + h, c:c + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # applying something
    while frame is not None:
        _, frame = vid_src.read()
        if frame is None:
            break

        # apply warping
        dst = map(
            lambda x: cv2.warpAffine(frame, x, (cols, rows)),
            MM
        )

        dst_sum = reduce(
            lambda a, b: np.add(a, b),
            dst
        )

        dst_avg = dst_sum / 8

        # camshift
        # backprojection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply meanshift to get the new location
        ret, track_window_new = cv2.CamShift(dst, track_window, term_crit)
        track_window = track_window_new

        # Draw it on image
        x, y, w, h = track_window

        cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)

        # showing
        cv2.imshow('img', frame)
        cv2.imshow('warp', dst_avg.astype('uint8'))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    vid_src.release()