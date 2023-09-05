#!/bin/bash

# Set the following variables
# The tensorboard logging will be creating in logs/<EXP>/<EXP_ID>
# The snapshots will be saved in snapshots/<EXP>/<EXP_ID>
EXP=v11_vos_mask
EXP_ID=v11_00_base

#
# No change are necessary starting here
#

SEED=32
CFG=configs/ytvos_mask.yaml
EXP_ID="YTM_${EXP_ID}"

CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $CURR_DIR/utils.bash

CMD="python train_mask.py --cfg $CFG --exp $EXP --run $EXP_ID --seed $SEED"
LOG_DIR=logs/${EXP}
LOG_FILE=$LOG_DIR/${EXP_ID}.log
echo "LOG: $LOG_FILE"

check_rundir $LOG_DIR $EXP_ID

NUM_THREADS=12

export OMP_NUM_THREADS=$NUM_THREADS
export MKL_NUM_THREADS=$NUM_THREADS

echo $CMD

CMD_FILE=$LOG_DIR/${EXP_ID}.cmd
echo $CMD > $CMD_FILE

#git rev-parse HEAD > $LOG_DIR/${EXP_ID}.head
#git diff > $LOG_DIR/${EXP_ID}.diff

nohup $CMD > $LOG_FILE 2>&1 &
sleep 1
tail -f $LOG_FILE
