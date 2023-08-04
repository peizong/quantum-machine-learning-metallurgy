#!/bin/bash

# https://stackoverflow.com/questions/1668649/how-to-keep-quotes-in-bash-arguments
args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="$args \"${i//\"/\\\"}\""
done

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
  export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

if [ -e /dev/nvidia0 ]; then nv="--nv"; fi

cmd=$(basename $0)

singularity exec $nv \
  --overlay /scratch/zp2137/cuQuantum/overlay-15GB-500K.ext3:ro \
  /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
  /bin/bash #-c "source /ext3/env.sh; $cmd $args"
