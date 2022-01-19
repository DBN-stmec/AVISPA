#!/bin/bash
TENSORFLOW_MODELS_DIR="/home/jonas/code/tensorflow-models"
FILE_PATH="$TENSORFLOW_MODELS_DIR/research/inception/inception/data/build_image_data.py"
DATA_DIR="/home/jonas/data/schneiden-werkzeug/cropped"
TRAIN_DIR="$DATA_DIR/train/"
VALIDATION_DIR="$DATA_DIR/validation/"
OUTPUT_DIR="../data/"
LABELS_FILE_PATH="../data/labels.txt"
TRAIN_SHARDS=64
VALIDATION_SHARDS=32

python $FILE_PATH --train_directory $TRAIN_DIR --validation_directory $VALIDATION_DIR --output_directory $OUTPUT_DIR --labels_file $LABELS_FILE_PATH --train_shards $TRAIN_SHARDS --validation_shards $VALIDATION_SHARDS