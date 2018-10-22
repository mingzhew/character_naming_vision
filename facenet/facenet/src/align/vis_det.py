import os
import sys
import cv2
import numpy as np
import pickle
from imageio import imsave

def vis_det(vidfile = ''):
	vidfile = '/scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/video_clips/tt0074285/tt0074285.sf-104189.ef-107667.video.mp4'
	trkdir = '/scratch/jiadeng_flux/mzwang/face_tracks_mtcnn/'
	vid = vidfile.split('/')[-1]
	mv = vidfile.split('/')[-2]
	trkfile = os.path.join(trkdir , mv , vid)+'.pi'
	with open(trkfile) as fid:
		bbox = pickle.load(fid)
	
	cap = cv2.VideoCapture(vidfile)
	idx = 0
	cnt = 0
	while cap.isOpened():
		ret , f = cap.read()
		if ret == True:
			if idx % 5 ==0:
				f = f[:,:,[2,1,0]].copy()
				for b in bbox[cnt]:
					cv2.rectangle(f , (int(b[0]) , int(b[1])) , (int(b[2]) , int(b[3])) , [255,0,0] , 2)
				imsave('output/%d.jpg'%(cnt) , f)
				cnt = cnt+1
		else: 
			break
		idx = idx+1 
