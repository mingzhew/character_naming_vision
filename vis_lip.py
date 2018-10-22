import cv2
import subprocess as sp
import os,sys
from imageio import imsave
import pickle
import numpy as np
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 
#trkdest = '/scratch/jiadeng_flux/mzwang/preprocess_naming/track/'
#lipdest = '/scratch/jiadeng_flux/mzwang/preprocess_naming/lip_motion'
trkdest = '/data/home/mzwang/workspace/movieQA/mqa_dataset/MovieQA_benchmark/story/preprocess/track'
lipdest = '/data/home/mzwang/workspace/movieQA/mqa_dataset/MovieQA_benchmark/story/preprocess/lip_motion_new'
vidfile = sys.argv[1]

vid = vidfile.split('/')[-1]
mv = vidfile.split('/')[-2]
with open(os.path.join(trkdest , mv , vid)+'.pi', 'rb') as fid:
	trks = pickle.load(fid)
with open(os.path.join(lipdest , mv , vid)+'.pi', 'rb') as fid:
	lips = pickle.load(fid, encoding='latin1')[1]

box_per_frame = []
lip_per_frame = []
cap = cv2.VideoCapture(vidfile)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(length):
	box_per_frame.append([])
	lip_per_frame.append([])
for j,trk in enumerate(trks):
	pos = trk[0]
	lpfeat = lips[j][:-1]-lips[j][1:]
	lpscore = np.abs(lpfeat).sum()
	for i,b in enumerate(trk[1]):
		box_per_frame[pos+i].append(b)
		lip_per_frame[pos+i].append(lpscore)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print (width, height)
a=b
#ffmpeg -r 24 -f image2 -s 406x720 -i ../output/%d.jpg -vcodec libx264 -crf 15 -pix_fmt yuv420p test.mp4
'''command = ['ffmpeg',
	'-y',
	'-r', '24',
	'-f', 'image2',
	'-s', '%dx%d'%(width,height),
	'-i', '-',
	'-an',
	'-vcodec', 'libx264',
	'-crf', '-15',
	'-pix_fmt', 'yuv420p',
	'vis_trk.mp4']
command = ['echo' , '-']
pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)
'''

cnt = 0
while cap.isOpened():
	ret , f = cap.read()
	if ret == True:
		_f = f[:,:,[2,1,0]].copy()
		for i,b in enumerate(box_per_frame[cnt]):
			cv2.rectangle(_f , (int(b[0]) , int(b[1])) , (int(b[2]) , int(b[3])) , [255,0,0] , 2)
			cv2.putText(_f , str(round(lip_per_frame[cnt][i],5)) , (int(b[0]),int(b[1]-3))  , cv2.FONT_HERSHEY_SIMPLEX, 1 , 255)	
		imsave('output/%d.jpg'%(cnt) , _f)
		#pipe.stdin.write( _f.tostring())
	else: break
	cnt = cnt+1

os.system('ffmpeg -y -r 24 -f image2 -s %dx%d -i'%(width,height)+' ../output/\%d.jpg -vcodec libx264 -crf 15 -pix_fmt yuv420p test.mp4')
