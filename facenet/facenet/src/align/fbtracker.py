import sys
import os
import numpy as np
sys.path.insert(0 , '/home/mzwang/libs/dlib/python_examples/')
import dlib
import glob
from skimage import io
import cv2
import pickle
import scipy.io as sio
from scipy.misc import imsave
def myfacetracker(vidfile , outfile , detfile):
# parameters
    print (detfile)
    track_length = 240
    track_th = 5
    PLOT = False
    minTrkLen = 10
    minDetScore = 0.98
    # load video
    #vidfile = '/scratch/jiadeng_fluxoe/mzwang/mvqa/MovieQA_benchmark/story/video_clips/tt0074285/tt0074285.sf-011382.ef-013071.video.mp4'
    #vidfile = '/scratch/jiadeng_fluxoe/mzwang/mvqa/MovieQA_benchmark/story/video_clips/tt0074285/tt0074285.sf-067882.ef-068429.video.mp4'
    cap = cv2.VideoCapture(vidfile)
    print (vidfile)
    print (cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    '''
    detector = dlib.get_frontal_face_detector()
    bbox = []
        num_det = 0
    length = 0
    while cap.isOpened():
        ret , f = cap.read()
        if ret == True:
            dets = detector(f , 1)
            bbox.append(dets)
            num_det = num_det+len(dets)
            length = length+1
            print len(dets)
        else:
            break
    '''
    #cap.release()
    #length = len(frame)
    # detection
    #print "detect %d faces"%(num_det)
    '''
    if PLOT:
        #fourcc = cv2.cv.CV_FOURCC('P','I','M','1')
        #out = cv2.VideoWriter('output.mpeg',fourcc, 24.0, frame[0].shape[:2])
            imgs = frame
               for i , img in enumerate(imgs):
                    for det in bbox[i]:
                            cv2.rectangle(img , (det.left() , det.top()) , (det.right() , det.bottom()) , [255,0,0] , 2)
                    #out.write(img)
                    imsave('output/detect%d.jpg'%(i), img)
    '''
    mv = vidfile.split('/')[-2]
    vid = vidfile.split('/')[-1]
    #trkdir = '/scratch/jiadeng_flux/mzwang/face_tracks_mtcnn'
    #trkfile = os.path.join(trkdir , mv , vid)+'.pi'
    with open(detfile, 'rb') as fid:
        bbox = pickle.load(fid)
    # tracking (delete duplicated bbox)
    ftracker = dlib.correlation_tracker()
    btracker = dlib.correlation_tracker()
    track = []
    frame = []
    box_per_frame = []
    for i in range(length):
        box_per_frame.append([])
    flag = np.ones([length , 1000])
    for i in range(len(bbox)):
        # load frames in buffer
        if i == 0:
            buffer_start = 0
            for j in range(1000):
                ret , f = cap.read()
                if ret == True:
                    frame.append(f)
                else:
                    break
        if i%100 == 50 and i >= 150:
            buffer_start = (i-50)*5    
            frame = frame[500:]
            for j in range(500):
                ret , f = cap.read()
                if ret == True:
                    frame.append(f)
                else: break
        # track each box
        for _i,_det in enumerate(bbox[i]):
            if flag[i][_i]: # det has not been overlap
                if _det[4] < minDetScore:
                    continue
                _det = _det[:4]
                t = [[_det[0],_det[1],_det[2],_det[3]]]
                pos = i*5
                _pos = pos
                bestFace = 0
                #tmplength = min(track_length , length - i)
                fidx = pos-buffer_start
                l = fidx-1 # backward index
                r = fidx+1 # forward index
                _det = dlib.drectangle(_det[0],_det[1],_det[2],_det[3])
                if fidx <= len(frame):
                    ftracker.start_track(frame[fidx] , _det)
                    btracker.start_track(frame[fidx] , _det)
                #for j in range(tmplength):
                    # forward tracking
                else:
                    break #stop tracking on this frame
                #calculate overlap
                def IoU(b1,b2):
                    if (b2[2]-b2[0])*(b1[2]-b1[0]) <= 0:
                        area = 0
                    elif (b2[3]-b2[1])*(b1[3]-b1[1]) <= 0:
                         area = 0
                    else:
                        maxb = [min(b1[0],b2[0]) , max(b1[1],b2[1]) , max(b1[2],b2[2]) , min(b1[3],b2[3])]
                        minb = [max(b1[0],b2[0]) , min(b1[1],b2[1]) , min(b1[2],b2[2]) , max(b1[3],b2[3])]
                        area = (minb[0]-minb[2])*(minb[1]-minb[3])
                        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
                        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
                        area = area/(a1+a2-area)
                    return area

                while r < len(frame):
                    score = ftracker.update(frame[r])
                    if score < track_th: break
                    else:
                        #print [r,ftracker.get_position()]
                        fdet = ftracker.get_position()
                        fdet = [fdet.left() , fdet.top() , fdet.right() , fdet.bottom()]
                        if r % 5 ==0:
                            _idx = int((r+buffer_start)/5)
                            for k in range(bbox[_idx].shape[0]):
                                if IoU(bbox[_idx][k] , fdet) > 0.5:
                                    flag[_idx][k] = 0
                                    fdet = bbox[_idx][k][:4]
                                    fdet = [fdet[0],fdet[1],fdet[2],fdet[3]]
                        t.append(fdet)
                        box_per_frame[r+buffer_start].append(fdet)
                    if r-_pos >= track_length: break
                    r = r+1
                while l >= 0:
                    score = btracker.update(frame[l])
                    if score < track_th: break
                    else:
                        #print [l,btracker.get_position()]
                        bdet = btracker.get_position()
                        bdet = [bdet.left() , bdet.top() , bdet.right() , bdet.bottom()]
                        _f = 1
                        for k in range(len(box_per_frame[l])):
                            if IoU(box_per_frame[l][k] , bdet) > 0.5:
                                _f = 0
                        if _f:
                            t.insert(0,bdet)
                            box_per_frame[l+buffer_start].append(bdet)
                            pos = pos-1#change start position
                            bestFace = bestFace+1
                    if _pos-l >= track_length: break
                    l = l-1
                if len(t) > minTrkLen:
                    track.append([pos , t , bestFace])
                    #print t
                    print ('face track starts from %d in %d frames'%(pos,len(t)))
    print ("get %d face tracks"%(len(track)))
    # store
        
    tracks = []
    for t in track:
        idx = t[0]
        pos = np.zeros((len(t[1]) , 4))
        best = t[2]
        for i,det in enumerate(t[1]):
            #print det
            pos[i] = det
        tracks.append([idx , pos , best])
    if len(outfile) > 0:
        with open(outfile , 'wb') as fid:
            pickle.dump(tracks , fid, protocol=2)
    return tracks
#plot 
    if PLOT:
        #fourcc = cv2.cv.CV_FOURCC('P','I','M','1')
        #out = cv2.VideoWriter('output.mpeg',fourcc, 24.0, frame[0].shape[:2])
        imgs = frame
        for _t in track:
            idx = _t[0]
            for i,det in enumerate(_t[1]):
                cv2.rectangle(imgs[idx+i] , (int(det.left()) , int(det.top())) , (int(det.right()) , int(det.bottom())) , [255,0,0] , 2)
            #out.write(img)
        for i,img in enumerate(imgs):
            imsave('output/track_%d.jpg'%(i), img)
