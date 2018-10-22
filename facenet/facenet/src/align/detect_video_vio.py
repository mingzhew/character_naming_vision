# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys,os
import tensorflow as tf
import detect_face
from scipy import misc
import pickle
from scipy.misc import imsave
#import cv2
import skvideo.io as vio

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        with tf.variable_scope('pnet'):
            data = tf.placeholder(tf.float32, (None,None,None,3), 'input')
            pnet = detect_face.PNet({'data':data})
            pnet.load('../../data/det1.npy', sess)
        with tf.variable_scope('rnet'):
            data = tf.placeholder(tf.float32, (None,24,24,3), 'input')
            rnet = detect_face.RNet({'data':data})
            rnet.load('../../data/det2.npy', sess)
        with tf.variable_scope('onet'):
            data = tf.placeholder(tf.float32, (None,48,48,3), 'input')
            onet = detect_face.ONet({'data':data})
            onet.load('../../data/det3.npy', sess)
        pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
        rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
        onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})

minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#source_path = '/home/david/datasets/casia/CASIA-maxpy-clean/0000045/002.jpg'
#img = misc.imread(source_path)

bbox = []
mvdir = sys.argv[1]
trkdir = '/scratch/jiadeng_flux/mzwang/face_tracks_mtcnn/'
if len(sys.argv) >= 3:
    trkdir = sys.argv[2]
mv = mvdir.split('/')[-1]
trkdest = os.path.join(trkdir , mv)
if os.path.exists(trkdest) == False:
    os.mkdir(trkdest)
vidlist = os.listdir(mvdir)
for vid in vidlist:
    bbox = []
    #vidfile = '/scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/video_clips/tt0095953/tt0095953.sf-109684.ef-111478.video.mp4'
    vidfile = os.path.join(mvdir , vid)
    trkfile = os.path.join(trkdest , vid)+'.pi'
    vdata = vio.vreader(vidfile)
    #cap = cv2.VideoCapture(vidfile)
    print (vidfile)
    #print (cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cnt = 0
    for f in vdata:
    #while cap.isOpened():
        #ret , f = cap.read()
        #if ret == True:
        if True:
            if cnt % 5 == 0:
                bounding_boxes, points = detect_face.detect_face(f, minsize, pnet_fun, rnet_fun, onet_fun, threshold, factor)
                print ('detect %f faces'%(bounding_boxes.shape[0]))
                bbox.append(bounding_boxes)
            cnt = cnt+1
        #else:
        #imsave('test%s.jpg'%(vid), f)
        #break
    with open(trkfile , 'w') as fid:
        pickle.dump(bbox , fid)
    #cap.release()
