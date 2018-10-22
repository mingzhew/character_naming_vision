import cv2
import os,sys
import numpy as np

fulldir = '/scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/full_videos/raw_videos/'
clipdir = '/scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/full_videos/clips/'
mv = sys.argv[1]
vidfile = os.path.join(fulldir , mv)+'.mp4'
print vidfile
cap = cv2.VideoCapture(vidfile)
vidlist = os.listdir(os.path.join(clipdir , mv))
endframe = []
startframe = []
length = []
order = []
for vid in vidlist:
	vidfile = os.path.join(clipdir , mv , vid)
	c = cv2.VideoCapture(vidfile)
	l = c.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
	lo = l
	ret , f = c.read()
	startframe.append(f)
	c.set(1,l-1)
	ret , f = c.read()
	'''
	while ret == False:
		l = l-1
		c.set(1,l-1)
		ret , f = c.read()
	'''
	print vid , ret , lo , l
	endframe.append(f)
	c.release()
	length.append(l)

ret , f = cap.read()
current = f
print ret
pos = 0
print len(vidlist)
for i in range(len(vidlist)):
	flag = 0
	poso = pos
	while flag == 0:
		for j in range(len(vidlist)):
			if np.abs(current - startframe[j]).sum() == 0:
				pos = poso+length[j]
				cap.set(1 , pos)
				ret , current = cap.read()
				order.append(j)
				print i,vidlist[j]
				flag = 1
				break
		if flag == 0:
			pos = pos-1
			cap.set(1 , pos)
			ret , current = cap.read()
target = ''
ft = ''
d2 = '/scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/full_videos/clips/'
d1 = '/scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/full_videos/clips/'
if os.path.exists(os.path.join(d2 , target , mv)) == False:
	os.mkdir(os.path.join(d2 , target , mv))
l = 1
for i in range(len(order)):
	r = l+length[order[i]]-1
	f1 = os.path.join(d1 , target , mv , vidlist[order[i]])+ft
	f2 = os.path.join(d2 , target , mv , '%d-%d.mp4'%(l , r))+ft
	l = r+1
	print 'mv %s %s'%(f1 , f2)
	os.system('mv %s %s'%(f1 , f2))
