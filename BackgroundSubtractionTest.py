from BackgroundSubtractionImpl import RunningAverageColor
from ObjectTrackingImpl import MeanShift2
from ObjectTrackingImpl import CamShift
import cv2
import numpy as np
import PostProcessing

__author__ = 'Luqman'

'''
test class for background subtraction
'''


class BackgroundSubtractionTest():
    def __init__(self, filename, algorithm, alg_params):
        self.algorithm = algorithm(alg_params)
        self.filename = filename

    def run(self):
        vid_src = cv2.VideoCapture(self.filename)
        _, frame = vid_src.read()
        object_box = None

        # applying background detection
        while frame is not None:
            _, frame = vid_src.read()
            if frame is None:
                break

            new_object_box, fg = self.algorithm.apply(frame, object_box)
            if object_box is not None:
                for box in object_box:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            object_box = new_object_box
            # showing
            cv2.imshow('img', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        vid_src.release()


class BackgroundSubtractionTestWithTracking():
    def __init__(self, filename, algorithm, alg_params):
        self.algorithm = algorithm(alg_params)
        self.filename = filename
        self.tracking = MeanShift2()

    def run(self):
        vid_src = cv2.VideoCapture(self.filename)
        _, frame = vid_src.read()
        prev_frame = None
        object_box = None

        # applying background detection
        while frame is not None:
            _, frame = vid_src.read()
            if frame is None:
                break

            new_object_box, fg = self.algorithm.apply(frame, object_box)

            if self.tracking.is_object_empty():
                if object_box is not None:
                    for box in object_box:
                        self.tracking.add_object(box, frame)
            else:
                # cek dalam new object box apakah ada yang baru # nanti
                if new_object_box is not None:
                    for new_box in new_object_box:
                        if PostProcessing.is_new_square(new_box, self.tracking.objects()):
                            self.tracking.add_object(new_box, frame)

            if object_box is not None:
                for box in object_box:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)

            if prev_frame is not None:
                self.tracking.run(prev_frame, frame)

            object_box = new_object_box
            # showing
            cv2.imshow('img', frame)

            prev_frame = np.copy(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        vid_src.release()

if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        video_src = sys.argv[1]
    else:
        video_src = 0

    test_running_average = BackgroundSubtractionTestWithTracking(video_src, RunningAverageColor, (0.02, 0.02))
    test_running_average.run()

# gimana caranya melokalisasi objek yang akan diamati