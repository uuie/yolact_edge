#!/bin/sh
mkdir /data
echo ${AWS_ACCESS_KEY_ID}:${AWS_SECRET_ACCESS_KEY} >/etc/passwd-s3fs 
chmod 600 /etc/passwd-s3fs
s3fs -o url=${MLFLOW_S3_ENDPOINT_URL} -o use_path_request_style -o allow_other -o ro -o passwd_file=/etc/passwd-s3fs -o umask=000 model-data /data
# setup user info
export LOGNAME=${FLOW_USER:-'flow_user'}
# normalize training id & data set
export TRAINING_SET=${TRAINING_SET:-'6'}
export EXPID=${EXPID:-'yolact_edge'}
# setup resource
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-'0'}
export NPROC_PER_NODE=$(echo ${CUDA_VISIBLE_DEVICES} |tr -s ',' '\n'|wc -l)
# start traning
python train.py --dataset bright_dataset --backbone_folder /data/backbones/ --batch_size 12