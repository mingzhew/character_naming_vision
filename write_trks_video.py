import cv2
from imageio import imsave, imread
import numpy as np
import os,sys
import pickle

def write_video(vidfile = '/scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/video_clips/tt0095953/tt0095953.sf-109684.ef-111478.video.mp4'):
	trkdest = '/scratch/jiadeng_flux/mzwang/face_tracks/'
	lipdest = '/scratch/jiadeng_fluxoe/mzwang/mvqa/lip_motion/'
	cap = cv2.VideoCapture(vidfile)
	vid = vidfile.split('/')[-1]
	mv = vidfile.split('/')[-2]
	trkfile = os.path.join(trkdest , mv , vid)+'.pi'
	trkfile = '/scratch/jiadeng_fluxoe/mzwang/naming/src/pyannote-video/build/test.pi'
	with open(trkfile) as fid:
		trks = pickle.load(fid)
	lipfile = os.path.join(lipdest , mv , vid)+'.pi'
	with open(lipfile) as fid:
		lipfeat = pickle.load(fid)[1]
	fs = cap.get(cv2.cv.CV_CAP_PROP_FPS)
	numframe = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	box_per_frame = {}
	trkidx_per_frame = {}
	for i in range(numframe):
	        box_per_frame[i] = []
	        trkidx_per_frame[i] = []
	for i,trk in enumerate(trks):
	        start = trk[0]
        	for j,box in enumerate(trk[1]):
                	pos = start+j
                	box_per_frame[pos].append(box)
                	trkidx_per_frame[pos].append(i)
	cnt = -1
	while cap.isOpened():
		ret , f = cap.read()
		if ret == False: break
		else:	
			cnt = cnt+1
			for i,box in enumerate(box_per_frame[cnt]):
				[x1,y1,x2,y2] = [int(round(box[0])),int(round(box[1])),int(round(box[2])),int(round(box[3]))]
				cv2.rectangle(f ,  (x1,y1) , (x2,y2) , [255,0,0] , 2)
			imsave('output/%d.jpg'%(cnt) , f)
