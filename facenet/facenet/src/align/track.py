import os,sys
import numpy as np
import pickle
from imageio import imread, imsave
from fbtracker import myfacetracker

mvdir = sys.argv[1]
mv = mvdir.split('/')[-1]
vidlist = os.listdir(mvdir)
#trkdest = '/scratch/jiadeng_flux/mzwang/face_tracks_mtcnn_trk'
trkdest = sys.argv[3]
detdest = sys.argv[2]
trkdir = os.path.join(trkdest , mv)
if os.path.exists(trkdir) == False:
	os.mkdir(trkdir)
for vid in vidlist:
	detfile = os.path.join(detdest , mv , vid)+'.pi'
	vidfile = os.path.join(mvdir , vid)
	outfile = os.path.join(trkdir , vid)+'.pi'
	trks = myfacetracker(vidfile , outfile , detfile)
