import cv2
from PointStatistics import PointValueScatter

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
    pvs = PointValueScatter()
    frame_count = 0

    # applying background detection
    while frame is not None:
        frame_count += 1
        _, frame = vid_src.read()
        if frame is not None:
            h, w, d = frame.shape
            pvs.fetch_point_value(frame, w/2, h/2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print w/2, h/2
    print frame_count
    pvs.plot_histogram()