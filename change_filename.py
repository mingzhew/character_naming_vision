import cv2
import os
import sys

mvdir = sys.argv[1]
vidlist = os.listdir(mvdir)
vidlist.sort()
l = 1
r = 0
for vid in vidlist:
	cap = cv2.VideoCapture(os.path.join(mvdir , vid))
	num = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	r = l+num-1
	cmd = 'mv %s/%s %s/%d-%d.mp4'%(mvdir , vid , mvdir , l , r)
	#print cmd
	os.system(cmd)
	l = l+num
	cap.release()
