#!/bin/bash

# uncompress the packed enviroment into the local dir
function prepare_env() {
  #mkdir -p env/
  #tar xf "afs/env/py3-torch-1.3.1.tgz" -C "env/"
  #export PYTHONPATH="$(pwd)/env:$PYTHONPATH"

  tar -xf "afs/liliulei/env/cuda11.tar"
  export PATH="$(pwd)/cuda11/bin:$PATH"
  export LD_LIBRARY_PATH="$(pwd)/cuda11/lib:$LD_LIBRARY_PATH"
}


function prepare_data() {
  START=`date +%s%N`;

  #cat afs/liliulei/datasets/youtube/train_all_frames.tar.* >train_all_frames.zip
  #unzip train_all_frames.zip
  #mv train_all_frames data/ytvos
  #unzip afs/liliulei/datasets/train.zip
  #mv train data/ytvos
  #unzip afs/liliulei/datasets/davis2017.zip
  #mv davis2017 data/
  tar -xf afs/liliulei/datasets/ytvalid2018.tar
  mv ytvos data/

  END=`date +%s%N`;
  time=$((END-START))
  time=`expr $time / 1000000000`
  echo "time for unzip dataset"
  echo $time

}


function prepare_model() {
  cp afs/liliulei/pretrain/resnet101_v1d.pth $(pwd)/resnet101_v1d.pth
  cp afs/liliulei/pretrain/iter_80000.pth $(pwd)/iter_80000.pth

}


# unify ui
prepare_env
prepare_data
#prepare_model

bash ./launch/infer_mask.sh ytvos
#bash ./launch/train_mask.sh
#bash ./launch/train.sh ytvos
#bash ./launch/gen_mask.sh
#tar -czvf pseudo_mask.tar pseudo_mask
#mkdir output
#mv pseudo_mask.tar output

