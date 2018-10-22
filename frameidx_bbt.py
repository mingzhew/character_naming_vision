import os
for i in range(6):
	with open('/home/mzwang/workspace/subtitle_naming/personID_CVPR2012/data_CVPR2012/facetracks/bbt_s01e0%d.facetrack'%(i+1)) as fid:
		cnt = 0
		f = open('../data/matidx/bbt%d.matidx'%(i+1) , 'w')
		for line in fid.readlines():
			cnt = cnt+1
			if cnt >= 3:
				s = line.split()
				f.write('%s %s\n'%(s[0],s[1]))
	
