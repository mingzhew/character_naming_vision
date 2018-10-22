#!/bin/bash

# create pbs file

jobName='naming_pre'
jobDict=`pwd`
pbsFile='scripts/feat.pbs'
node=1
ppn=3
gpus=1
mem='4096'
runtime='01-00:00:00'
logfile='mvqa_pre.log'
i=1
#for i in `seq 1 9`;
while IFS='' read -r line || [[ -n "$line" ]]; do
#for vid in /scratch/jiadeng_flux/mzwang/MovieQA_benchmark/story/full_videos/clips/pirate/*; do
	fn=$pbsFile${i}
    echo -e '#!/bin/bash' > $fn
    echo '#' >> $fn
    echo '#SBATCH --job-name='$jobName >> $fn
    echo '#SBATCH --output='$jobName$i'_output' >> $fn
    echo '#' >> $fn
    echo '#SBATCH --mail-user=mzwang@umich.edu' >> $fn
    echo -e '#SBATCH --mail-type=ALL\n#' >> $fn
    echo -e '#SBATCH --nodes=1\n#SBATCH --ntasks=1' >> $fn
#echo '#SBATCH -p vl-fb-gtx1080' >> $fn
    echo '#SBATCH --cpus-per-task='$ppn >> $fn
    echo '#SBATCH --mem-per-cpu='$mem >> $fn
    echo '#SBATCH --gres=gpu:'$gpus >> $fn
    echo '#SBATCH --time='$runtime >> $fn
    echo '#' >> $fn
    echo '#SBATCH --workdir='$jobDict >> $fn
    # exclude some nodes
    echo '#SBATCH --exclude=compute-2,compute-5' >> $fn

	echo './preprocess.sh '$line >> $fn
#sbatch $fn
	i=$((i+1))	
done < "$1"

