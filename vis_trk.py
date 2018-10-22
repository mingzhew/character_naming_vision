import cv2
import subprocess as sp
import os,sys
from imageio import imsave
import pickle
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL) 
trkdest = '/scratch/jiadeng_flux/mzwang/preprocess_naming/track/'
vidfile = sys.argv[1]

vid = vidfile.split('/')[-1]
mv = vidfile.split('/')[-2]
with open(os.path.join(trkdest , mv , vid)+'.pi') as fid:
	trks = pickle.load(fid)

box_per_frame = []
cap = cv2.VideoCapture(vidfile)
length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
for i in range(length):
	box_per_frame.append([])
for trk in trks:
	pos = trk[0]
	for i,b in enumerate(trk[1]):
		box_per_frame[pos+i].append(b)
width = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
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
'''
cnt = 0
while cap.isOpened():
	ret , f = cap.read()
	if ret == True:
		_f = f[:,:,[2,1,0]].copy()
		for b in box_per_frame[cnt]:
			cv2.rectangle(_f , (int(b[0]) , int(b[1])) , (int(b[2]) , int(b[3])) , [255,0,0] , 2)
		imsave('../output/%d.jpg'%(cnt) , _f)
		#pipe.stdin.write( _f.tostring())
	else: break
	cnt = cnt+1
'''
os.system('ffmpeg -y -r 24 -f image2 -s %dx%d -i'%(width,height)+' ../output/\%d.jpg -vcodec libx264 -crf 15 -pix_fmt yuv420p test.mp4')
