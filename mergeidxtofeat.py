import numpy as np
import h5py
import os

facedir = '/scratch/jiadeng_fluxoe/mzwang/mvqa/cropimg_all'
featdir = '/scratch/jiadeng_fluxoe/mzwang/mvqa/vggfacefeat'
outdir = '/scratch/jiadeng_fluxoe/mzwang/mvqa/vggfacefeat_idx'
if os.path.exists(outdir) == False:
	os.mkdir(outdir)
with open('file.txt') as fid:
	mvlist = fid.readlines()
for mv in mvlist:
	mv = mv.strip()
	vidlist = os.listdir(os.path.join(featdir , mv))
	if os.path.exists(os.path.join(outdir , mv)) == False:
		os.mkdir(os.path.join(outdir , mv))
	for vid in vidlist:
		with h5py.File(os.path.join(facedir , mv , vid)) as hf:
			idx = hf['idx'][:]
		with h5py.File(os.path.join(featdir , mv , vid)) as hf:
			feat = hf['fc7'][:]
		with h5py.File(os.path.join(outdir , mv , vid)) as hf:
			hf.create_dataset('idx' , data=idx)
			hf.create_dataset('fc7' , data=feat)
		print vid
