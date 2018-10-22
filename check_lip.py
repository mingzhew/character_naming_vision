import os
import sys

datadir='/home/mzwang/workspace/movieQA/mqa_dataset/MovieQA_benchmark/story'
clip_dir=datadir+'/video_clips'
det_dir=datadir+'/preprocess/detect'
trk_dir=datadir+'/preprocess/track'
lip_dir=datadir+'/preprocess/lip_motion'
landmark_dir=datadir+'/preprocess/landmark'
crop_dir=datadir+'/preprocess/cropimg'
feat_dir=datadir+'/preprocess/feature'
matidx_dir=datadir+'/preprocess/matidx'

mvlist = os.listdir(lip_dir)
for mv in mvlist:
    l1 = os.listdir(os.path.join(clip_dir, mv))
    l2 = os.listdir(os.path.join(lip_dir, mv))
    for vid in l1:
        if vid+'.pi' not in l2:
            # $vid $trk_dir $landmark_dir $lip_dir $crop_dir
            print (mv, vid, len(l1), len(l2))
            vidp = os.path.join(clip_dir, mv, vid)
            cmd = 'python crop_lip.py %s %s %s %s %s' % (vidp, trk_dir, landmark_dir, lip_dir, crop_dir)
            #print (cmd)
            os.system(cmd)


