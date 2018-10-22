import numpy as np
import cv2
import sys,os

fn = sys.argv[1]
print fn
mv = sys.argv[3]
outdest = sys.argv[4]
vid_len = 5
vid = cv2.VideoCapture(fn)
numframe = vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
print numframe
fs = vid.get(cv2.cv.CV_CAP_PROP_FPS)
matidx_dir = sys.argv[2]
#fs = int(round(fs))
vid_frame = vid_len * 60 * int(round(fs))
matidx = []
fid = open(os.path.join(matidx_dir , mv+'.matidx') , 'w')
for i in range(int(numframe)):
	matidx.append(round(i/fs , 3))
	fid.write('%d %.3f\n'%(i , matidx[i]))
fid.close()
fid = open('cut_%s.sh'%(mv) , 'w')
l = 1
outdir = os.path.join(outdest , mv)
if os.path.exists(outdir) == False:
	os.mkdir(outdir)
while l < numframe:
	r = min(l+vid_frame-1,int(numframe)-1)
	outfn = '%s/%d-%d.mp4'%(outdir , l , r)
	fid.write('ffmpeg -i %s -ss %.3f -t %.3f %s\n'%(fn , matidx[l] , matidx[r]-matidx[l] , outfn))
	l = l + vid_frame
fid.close()
