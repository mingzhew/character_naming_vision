#!/bin/bash
# global file path
datadir='/home/mzwang/workspace/movieQA/mqa_dataset/MovieQA_benchmark/story'
#local_dir='/scratch/jiadeng_fluxoe/mzwang/preporcess_naming'
local_dir=`pwd`
full_movie_dir='/scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/full_videos/raw_videos'
clip_dir=$datadir'/video_clips'
det_dir=$datadir'/preprocess/detect'
trk_dir=$datadir'/preprocess/track'
lip_dir=$datadir'/preprocess/lip_motion'
landmark_dir=$datadir'/preprocess/landmark'
crop_dir=$datadir'/preprocess/cropimg'
feat_dir=$datadir'/preprocess/feature'
matidx_dir=$datadir'/preprocess/matidx'
mv=$1

# load modules
export PATH="/home/mzwang/anaconda3_new/bin:$PATH"
source activate py34
module load ffmpeg
export PYTHONPATH="/data/opt/opencv/3.2.0/lib/python3.4/site-packages/:$PYTHONPATH"
export PYTHONPATH="/data/opt/opencv/3.2.0/lib:$PYTHONPATH"


# 1 cut video into 5 mins
python generate_cut_command.py $full_movie_dir/$mv/video.mp4 $matidx_dir $mv $clip_dir
chmod +x cut_$mv.sh
./cut_$mv.sh
if [ -f "$full_movie_dir/$mv.mp4" ]; then
	if [ -d "$clip_dir/$mv" ]; then 
		mkdir $clip_dir/$mv
	fi
    #python generate_cut_command.py $full_movie_dir/$mv.mp4 $matidx_dir $mv $clip_dir
    #ffmpeg -i $full_movie_dir/$mv.mp4 -c copy -f segment -segment_time 300 -reset_timestamps 1 $clip_dir/$mv/%03d.mp4
    python change_filename.py $clip_dir/$mv

	# 2 detect with mtcnn
    cd facenet/facenet/src/align
    python detect_video.py $clip_dir/$mv $det_dir > log_$mv
    python track.py $clip_dir/$mv $det_dir $trk_dir
	cd $local_dir
	
	# 4 crop & lip
    for vid in $clip_dir/$mv/*; do
		python crop_lip.py $vid $trk_dir $landmark_dir $lip_dir $crop_dir
	done
	# 5 features
    #nvidia-smi
    for vid in $clip_dir/$mv/*; do
    	th extract_feat_batch.lua -vid $vid -crop $crop_dir -feat $feat_dir
	done
fi
