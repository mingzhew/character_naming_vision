import numpy as np
import cv2
import os
import sys
from imageio import imsave
import pickle
from PIL import Image
import h5py

viddir = '/scratch/jiadeng_fluxoe/mzwang/mvqa/TBBT_MP4/video'
trksdir = '/scratch/jiadeng_flux/mzwang/face_tracks/bbt'
annodir = '/home/mzwang/workspace/subtitle_naming/personID_CVPR2012/data_CVPR2012/facetracks'

if os.path.exists(trksdir) == False:
	os.mkdir(trksdir)
vidlist = os.listdir(viddir)
vidlist.sort()
print vidlist
frames = []
for vid in vidlist:
	print vid
	fntrk = 'bbt_s01e0%s.facetrack'%(vid[0])
	with open(os.path.join(annodir , fntrk)) as fid:
		raw_data = fid.readlines()
	frame = []
	cnt = 0
	cap = cv2.VideoCapture(os.path.join(viddir , vid))
	while cap.isOpened():
		ret , f = cap.read()
		if ret == True:                
			frame.append(f)
			anno = raw_data[cnt+2].split()
			if len(anno) >= 2:
				print anno
				for i in range(int(anno[2])):
					pos = i*9+3
					box = [int(anno[pos+1]),int(anno[pos+2]),int(anno[pos+3]),int(anno[pos+4])]
					box[2] = box[2]+box[0]
					box[3] = box[3]+box[1]
					print box
					cv2.rectangle(f , (box[0],box[1]) , (box[2],box[3]) , [255,0,0] , 2)
			imsave('output/%s_%d.jpg'%(vid[0] , cnt) , f)
			cnt = cnt+1
			if cnt == 200:
				break
		else:
			break
	cap.release()
	frames.append(frame)
	break	
