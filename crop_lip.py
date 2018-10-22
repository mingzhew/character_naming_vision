import sys
import os
import numpy as np
sys.path.insert(0 , '/home/mzwang/libs/dlib/python_examples')
import dlib
import glob
from skimage import io
import cv2
import h5py
import pickle
import scipy.io as sio
from scipy.misc import imsave
from scipy.signal import butter, lfilter
from scipy.signal import freqz
import math
from PIL import Image

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

drawpic = False
outpath = 'output'
lowcut = 3.75
highcut = 7.5
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
vidpath = sys.argv[1] # vid path
vid = vidpath.split('/')[-1]
mv = vidpath.split('/')[-2]
trkdir = os.path.join(sys.argv[2] , mv)
dest_landmark = os.path.join(sys.argv[3] , mv)
dest_lipmotion = os.path.join(sys.argv[4] , mv)
dest_faces = os.path.join(sys.argv[5] , mv)
dest_ld = os.path.join(dest_landmark , vid)+'.pi'
dest_lm = os.path.join(dest_lipmotion , vid)+'.pi'
dest_fi = os.path.join(dest_faces , vid)+'.h5'
if os.path.exists(dest_landmark) == False:
    os.mkdir(dest_landmark)
if os.path.exists(dest_lipmotion) == False:
    os.mkdir(dest_lipmotion)
if os.path.exists(dest_faces) == False:
    os.mkdir(dest_faces)
video = vidpath
trkfile = os.path.join(trkdir , vid)+'.pi'
trksldmk = []
trkslpmt = []
trkslpmt_bpf = []
cap = cv2.VideoCapture(video)
fs = cap.get(cv2.CAP_PROP_FPS)
print (vid)
numframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print (numframe)
box_per_frame = {}
trkidx_per_frame = {}
for i in range(numframe):
    box_per_frame[i] = []
    trkidx_per_frame[i] = []
with open(trkfile, 'rb') as fid:
    trks = pickle.load(fid)
#reverse index
for i,trk in enumerate(trks):
    start = trk[0]
    for j,box in enumerate(trk[1]):
        pos = start+j
        box_per_frame[pos].append(box)
        trkidx_per_frame[pos].append(i)
trksldmk = []
trkslpmt = []
trkslpmt_bpf = []
mvidx = np.ndarray(len(trks))
mvidx_sum = np.ndarray(len(trks))
_sum = 0
for i,trk in enumerate(trks):
    trksldmk.append([])
    trkslpmt.append(np.ndarray((len(trk[1]))))
    trkslpmt_bpf.append(np.ndarray((len(trk[1]))))
    mvidx[i] = int((len(trk[1])-1)/5+1)
    mvidx_sum[i] = _sum
    _sum = _sum+mvidx[i]
mvfaces = np.ndarray((int(_sum),224,224,3))
cnt = -1
while cap.isOpened():
    ret , f = cap.read()
    if ret == False:
        break
    else:
        # a legal frame. crop faces and detect landmark, calculate lipmotion for this frame. store lipmotion with reverse index.
        cnt = cnt+1
        for i,box in enumerate(box_per_frame[cnt]):
            trkidx = trkidx_per_frame[cnt][i]
            shape = predictor(f , dlib.rectangle(int(box[0]),int(box[1]),int(box[2]),int(box[3])))
            lippos = [shape.part(62),shape.part(66),shape.part(51),shape.part(57)]
            trksldmk[trkidx].append(shape)
            boxnum = cnt-trks[trkidx][0]
            trkslpmt[trkidx][boxnum] = math.sqrt((lippos[1].x-lippos[0].x)**2+(lippos[1].y-lippos[0].y)**2)/(box[3]-box[1])
            if boxnum % 5 == 0:
                img = Image.fromarray(f)
                print (box)
                [x1,y1,x2,y2] = [int(round(box[0])),int(round(box[1])),int(round(box[2])),int(round(box[3]))]
                x1 = int(max(x1, 0))
                y1 = int(max(y1, 0))
                x1 = int(min(x1, f.shape[1]-2))
                y1 = int(min(y1, f.shape[1]-2))
                x2 = int(max(x2, 1))
                y2 = int(max(y2, 1))
                x2 = int(min(x2, f.shape[1]-1))
                y2 = int(min(y2, f.shape[0]-1))
                print (x1, y1, x2, y2)
                face = img.crop((x1,y1,x2,y2))
                face = face.resize((224,224))
                numfaces = int(mvidx_sum[trkidx]+boxnum/5)
                mvfaces[numfaces,:,:,:] = np.array(face.getdata()).reshape(face.size[0],face.size[1],3)    
                #imsave('output/%s.jpg'%(numfaces) , mvfaces[numfaces,:,:,:])
cap.release()            
for i,trk in enumerate(trks):
    feat = butter_bandpass_filter(trkslpmt[i] , lowcut , highcut , fs , order=3)
    trkslpmt_bpf[i] = feat        

with open(dest_ld,'wb') as fid:
    pickle.dump(trksldmk , fid, protocol=2)
with open(dest_lm,'wb') as fid:
    pickle.dump([trkslpmt , trkslpmt_bpf] , fid, protocol=2)
with h5py.File(dest_fi , 'w') as hf:
    hf.create_dataset('faces' , data=mvfaces)
    hf.create_dataset('idx' , data=mvidx)




