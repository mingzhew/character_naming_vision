import pickle
import bisect
import os
import sys
import h5py
import numpy as np
# input: movie id. two timestamp in seconds, such as 123.456 , 987.654
# output: face tracks inside this timestamp and talking scores

#base_path = '/local/mazab/movieQA/annotated_subtitles/video/' 
base_path = '/scratch/jiadeng_fluxoe/mzwang/mvqa/'
trk_path = '/scratch/jiadeng_flux/mzwang/'
matpath = '/scratch/jiadeng_fluxoe/mzwang/naming/'


def face_score(mv , t1 , t2):
	minlen = 20 # minium length of face tracks to be considered.
	lpdir =  base_path + 'lip_motion'
	idxdir = matpath + 'data/matidx'
	trkdir = trk_path + 'face_tracks'
	featdir =base_path + 'vggfacefeat_idx'
	#genderdir = base_path + 'gender'
	mvdir = os.path.join(featdir , mv)
	vidlist = os.listdir(mvdir)
        if mv == 'tt0118971': 
            for v in vidlist: print v
	vidstart = []
	vidend = []
	if mv[0] == 't':
		for vid in vidlist:
			a = vid.find('-')
			b = vid.find('.',a)
			vidstart.append(int(vid[a+1:b]))
			a = vid.find('-',b)
			b = vid.find('.',a)
			vidend.append(int(vid[a+1:b]))
	elif mv[:3] == 'bbt':
                vidstart.append(0)
                vidend.append(100000)
        else:
                for vid in vidlist:
                        a = vid.find('-')
                        b = vid.find('.',a)
                        vidstart.append(int(vid[:a]))
                        vidend.append(int(vid[a+1:b]))

	timeidx = []
	with open(os.path.join(idxdir , mv+'.matidx')) as fid:
		for line in fid.readlines():
			timeidx.append(line.strip().split())
	timesec = [float(e[1]) for e in timeidx]
	i1 = bisect.bisect(timesec , t1)
	i2 = bisect.bisect(timesec , t2)
        if i2 >= len(timeidx):
                return []
	f1 = float(timeidx[i1][0])
	f2 = float(timeidx[i2][0])
	targettrks = []
	for i in range(len(vidlist)):
		if vidstart[i] <= f1 and vidend[i] >= f2:
			# find the target video
			tt1 = f1 - vidstart[i]
			tt2 = f2 - vidstart[i]
			lpfile = os.path.join(lpdir , mv , vidlist[i][:-2]+'pi')
			trkfile = os.path.join(trkdir , mv , vidlist[i][:-2]+'pi')
			featfile = os.path.join(featdir , mv , vidlist[i])
			#genderfile = os.path.join(genderdir , mv , vidlist[i])
			#facefile = os.path.join('/scratch/jiadeng_fluxoe/mzwang/mvqa/cropimg_all' , mv , vidlist[i])
			with open(lpfile) as fid:
				lpfeat = pickle.load(fid)[1]
			with open(trkfile) as fid:
				trks = pickle.load(fid)	
			with h5py.File(featfile , 'r') as fid:
				vggfeat = fid['fc7'][:]
				idx = fid['idx'][:]
	#		with h5py.File(genderfile , 'r') as fid:
	#			gender = fid['gender'][:]
			for j,trk in enumerate(trks):
				start = trk[0]
				length = len(trk[1])
				end = start+length
				if min(end , tt2) - max(start , tt1) > minlen:
					# a target face tracks
					facescore = []
					if j == 0:
						idxstart = 0
					else:
						idxstart = np.sum(idx[:j])	
					idxend = np.sum(idx[:j+1])
					facescore.append(vggfeat[idxstart:idxend].mean(0))
					lpstart = max(0,tt1-start)
					lpend = min(0 , tt2-end)+len(trk[1])
					_lpfeat = lpfeat[j][lpstart:lpend-1]-lpfeat[j][lpstart+1:lpend]
					talkingscore = np.abs(_lpfeat).sum()
					facescore.append(talkingscore)
	#				facescore.append(gender[j])
					#facescore.append(faces[idxstart])
					targettrks.append(facescore)
	return targettrks


#if __name__=='__main__':
#    print face_score('tt0074285',12,15)
