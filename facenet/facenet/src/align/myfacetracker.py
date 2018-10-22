import sys
import os
import numpy as np
import dlib
import glob
from skimage import io
import cv2
#import skvideo.io as vio
import pickle
import scipy.io as sio
from scipy.misc import imsave
def myfacetracker(vidfile , outfile):
# parameters
    track_length = 75
    track_th = 5
    PLOT = False
    # load video
    #vidfile = '/scratch/jiadeng_fluxoe/mzwang/mvqa/MovieQA_benchmark/story/video_clips/tt0074285/tt0074285.sf-011382.ef-013071.video.mp4'
    #vidfile = '/scratch/jiadeng_fluxoe/mzwang/mvqa/MovieQA_benchmark/story/video_clips/tt0074285/tt0074285.sf-067882.ef-068429.video.mp4'
    cap = cv2.VideoCapture(vidfile)
    print vidfile
    print cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
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
    #cap.release()
    #length = len(frame)
    # detection
    print "detect %d faces"%(num_det)
    if PLOT:
        #fourcc = cv2.cv.CV_FOURCC('P','I','M','1')
        #out = cv2.VideoWriter('output.mpeg',fourcc, 24.0, frame[0].shape[:2])
        imgs = frame
           for i , img in enumerate(imgs):
                for det in bbox[i]:
                cv2.rectangle(img , (det.left() , det.top()) , (det.right() , det.bottom()) , [255,0,0] , 2)
                #out.write(img)
                imsave('output/detect%d.jpg'%(i), img)
    
    #with open(outfile+'_det', 'w') as fid:
    #    pickle.dump(bbox , fid)
    # tracking (delete duplicated bbox)
    tracker = dlib.correlation_tracker()
    track = []
    flag = np.ones([length , 1000])
    for i in range(length):
        for _i,_det in enumerate(bbox[i]):
            if flag[i][_i]:
                t = [_det]
                tmplength = min(track_length , length - i)
                for j in range(tmplength):
                    if j == 0:
                        cap.set(1,i)
                        ret , f = cap.read()
                        if ret == False:
                            break
                        tracker.start_track(f , _det)
                    else:
                        ret , f = cap.read()
                        if ret == False:
                            break
                        f1 = f.copy()
                        score = tracker.update(f1)
                        if score < track_th:
                            break
                        else:
                            print [j,tracker.get_position()]
                            t.append(tracker.get_position())
                            new_det = tracker.get_position()
                            l = len(bbox[i+j])
                            for k in range(l):
                                area = 0
                                det = bbox[i+j][k]
                                b1 = [det.left() , det.top() , det.right() , det.bottom()]
                                b2 = [new_det.left() , new_det.top() , new_det.right() , new_det.bottom()]
                                if (b2[2]-b2[0])*(b1[2]-b1[0]) <= 0:
                                    area = 0
                                elif (b2[3]-b2[1])*(b1[3]-b1[1]) <= 0:
                                    area = 0
                                else:
                                    maxb = [min(b1[0],b2[0]) , max(b1[1],b2[1]) , max(b1[2],b2[2]) , min(b1[3],b2[3])]
                                    minb = [max(b1[0],b2[0]) , min(b1[1],b2[1]) , min(b1[2],b2[2]) , max(b1[3],b2[3])]
                                    area = (minb[0]-minb[2])*(minb[1]-minb[3])/(maxb[0]-maxb[2])/(minb[1]-minb[3])
                                if area > 0.5:
                                    flag[i+j][k] = 0
                if len(t) > 0:
                    track.append([i,t])
    print "get %d face tracks"%(len(track))
    # store
    tracks = []
    for t in track:
        idx = t[0]
        pos = np.zeros((len(t[1]) , 4))
        for i,det in enumerate(t[1]):
            pos[i] = [det.left() , det.top() , det.right() , det.bottom()]
        tracks.append([idx , pos])
    with open(outfile , 'wb') as fid:
        pickle.dump(tracks , fid, protocol=2)
    
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
