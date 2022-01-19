#!/bin/bash
MODEL=$1
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="$SCRIPTS_DIR/../models/darknet"
MODEL_DIR=$MODELS_DIR/$MODEL
DARKNET_DIR="/media/jonas/data/code/darknet/"
DATA_DIR="$SCRIPTS_DIR/../data/darknet"
DATA_PATH=$MODEL_DIR/$MODEL.data
CONFIG_PATH=$MODEL_DIR/$MODEL.cfg
BACKUP_DIR=$MODEL_DIR/backup
BACKUP_PATH=$DATA_DIR/darknet19_448.conv.23
LOG_DIR=$MODEL_DIR/logs
LOG_PATH=$LOG_DIR/$MODEL.log

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

mkdir -p $LOG_DIR
mkdir -p $BACKUP_DIR

cd $DARKNET_DIR

./darknet detector train $DATA_PATH $CONFIG_PATH $BACKUP_PATH $BACKUP_PATH >> $LOG_PATH