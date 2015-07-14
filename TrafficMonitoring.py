import sys
from BackgroundSubtractionImpl import FrameDifference
from BackgroundSubtractionImpl import RunningAverage
from BackgroundSubtractionImpl import MedianRecursive
from BackgroundSubtractionImpl import OnlineKMeans
from BackgroundSubtractionImpl import SingleGaussian
from BackgroundSubtractionImpl import KDE

vid_file = ''
if len(sys.argv) < 2:
    print 'Masukkan nama file video!'
    sys.exit(1)
else:
    vid_file = sys.argv[1]

print "Input video: ", vid_file

back = RunningAverage(vid_file, 0.01)
back.run()

# back2 = RunningAverage(vid_file, 0.015)
# back2.run()

back3 = OnlineKMeans(vid_file, 0.02)
back3.run()
#
# back0 = SingleGaussian(vid_file, 0.95, 0.01)
# back0.run()
#
# back99 = KDE(vid_file, 0.9, 0.05, 30)
# back99.run()