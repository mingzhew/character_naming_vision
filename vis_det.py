import cv2
import os,sys
from imageio import imsave
import pickle

trkdest = '/scratch/jiadeng_flux/mzwang/preprocess_naming/detect/'
vidfile = sys.argv[1]

vid = vidfile.split('/')[-1]
mv = vidfile.split('/')[-2]
with open(os.path.join(trkdest , mv , vid)+'.pi') as fid:
	dets = pickle.load(fid)

cap = cv2.VideoCapture(vidfile)
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
cnt = 0
while cap.isOpened():
	ret , f = cap.read()
	if ret == True:
		if cnt % 5 == 0:
			_f = f[:,:,[2,1,0]].copy()
			for b in dets[int(cnt/5)]:
				cv2.rectangle(_f , (int(b[0]) , int(b[1])) , (int(b[2]) , int(b[3])) , [255,0,0] , 2)
				cv2.putText(_f , str(b[4]) , (int(b[0]),int(b[1]-3))  , cv2.FONT_HERSHEY_SIMPLEX, 1 , 255)
			imsave('../output/%d.jpg'%(cnt) , _f)
	else: break
	cnt = cnt+1
