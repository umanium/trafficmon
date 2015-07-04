import sys
from BackgroundSubtraction import FrameDifference
from BackgroundSubtraction import RunningAverage
from BackgroundSubtraction import MedianRecursive
from BackgroundSubtraction import OnlineKMeans
from BackgroundSubtraction import SingleGaussian
from BackgroundSubtraction import KDE

vidFile = ''
if(len(sys.argv) < 2):
    print 'Masukkan nama file video!'
    sys.exit(1)
else:
    vidFile = sys.argv[1]

print "Input video: ", vidFile

back = RunningAverage(vidFile, 0.015)
back.run()

# back2 = MedianRecursive(vidFile)
# back2.run()

back3 = OnlineKMeans(vidFile, 0.02)
back3.run()

back0 = SingleGaussian(vidFile, 0.9, 0.01)
back0.run()

back99 = KDE(vidFile, 0.9, 0.01, 50)