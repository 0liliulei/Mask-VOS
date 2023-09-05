#!/bin/bash

#
# Arguments
#

# suffix for the output directory (see below)
VER=v01

# defines the path to the snapshot (see below)
EXP=v03_vos
RUN_ID=YT_v03_00_base

# SNAPSHOT name. Note .pth will be attached
# See README.md to download these snapshots

SNAPSHOT=epoch120_score0.675_key   # YouTube-VOS
#SNAPSHOT=trackingnet_e088_res4 # TrackingNet
#SNAPSHOT=oxuva_e430_res4       # OxUvA
#SNAPSHOT=kinetics_e026_res4    # Kinetics

# codename of the final output layer [res3|res4|key]
KEY=key

FILELIST=filelists/gen_mask_ytvos2018_train

#
# Changing the following is not necessary
#

# The config file is irrelevant
# since the inference parameters are
# always the same
CONFIG=configs/ytvos.yaml
OUTPUT_DIR=/data02/liliulei/pseudo_mask

EXTRA="$EXTRA --seed 0 --set TEST.KEY $KEY"
SAVE_ID=${RUN_ID}_${SNAPSHOT}_${VER}

SNAPSHOT_PATH=${SNAPSHOT}.pth
if [ ! -f $SNAPSHOT_PATH ]; then
  echo "Snapshot $SNAPSHOT_PATH NOT found."
  exit 1;
fi


#
# Code goes here
#
LISTNAME=`basename $FILELIST .txt`
SAVE_DIR=$OUTPUT_DIR/$EXP/$SAVE_ID/$LISTNAME
LOG_FILE=$OUTPUT_DIR/$EXP/$SAVE_ID/${LISTNAME}_${KEY}.log

NUM_THREADS=12
export OMP_NUM_THREADS=$NUM_THREADS
export MKL_NUM_THREADS=$NUM_THREADS


CMD="python gen_mask.py   --cfg $CONFIG \
                           --exp $EXP \
                           --run $RUN_ID \
                           --resume $SNAPSHOT_PATH \
                           --infer-list $FILELIST \
                           --mask-output-dir $SAVE_DIR \
                           $EXTRA"

if [ ! -d $SAVE_DIR ]; then
  echo "Creating directory: $SAVE_DIR"
  mkdir -p $SAVE_DIR
else
  echo "Saving to: $SAVE_DIR"
fi

echo $CMD > ${SAVE_DIR}.cmd

echo $CMD
nohup $CMD > $LOG_FILE 2>&1 &

sleep 1
tail -f $LOG_FILE
