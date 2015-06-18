import sys
from BackgroundSubtraction import RunningAverage
from BackgroundSubtraction import MedianRecursive
from BackgroundSubtraction import OnlineKMeans

vidFile = ''
if(len(sys.argv) < 2):
    print 'Masukkan nama file video!'
    sys.exit(1)
else:
    vidFile = sys.argv[1]

print "Input video: ", vidFile
back = RunningAverage(vidFile, 0.02)
back.run()

back2 = MedianRecursive(vidFile)
back2.run()

back3 = OnlineKMeans(vidFile, 0.02)
back3.run()