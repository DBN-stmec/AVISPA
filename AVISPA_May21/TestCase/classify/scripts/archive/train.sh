#!/bin/bash
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script performs the following operations:
# 1. Downloads the Flowers dataset
# 2. Fine-tunes an InceptionV3 model on the Flowers training set.
# 3. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v3_on_flowers.sh
set -e

SLIM_DIR="/home/jonas/code/tensorflow-models/research/slim/"
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/.."
MODEL_NAME="inception_v3"
DATASET_NAME="schneide"
TRAIN=false
EVAL=false
EXPORT=true

# Where the pre-trained InceptionV3 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR="${BASE_DIR}/pretrained_models/${MODEL_NAME}"

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR="${BASE_DIR}/train/${MODEL_NAME}"

# Where the dataset is saved to.
DATASET_DIR="${BASE_DIR}/data"

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/model.ckpt ]; then
  wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
  tar -xvf inception_v3_2016_08_28.tar.gz
  mv inception_v3.ckpt ${PRETRAINED_CHECKPOINT_DIR}/model.ckpt
  rm inception_v3_2016_08_28.tar.gz
fi

# Download the dataset
: '
python download_and_convert_data.py \
  --dataset_name=flowers \
  --dataset_dir=${DATASET_DIR}
'
# Fine-tune only the new layers for 1000 steps.

if [ "$TRAIN" = true ]; then
    python ${SLIM_DIR}train_image_classifier.py \
      --train_dir=${TRAIN_DIR} \
      --dataset_name=${DATASET_NAME} \
      --dataset_split_name=train \
      --dataset_dir=${DATASET_DIR} \
      --model_name=${MODEL_NAME} \
      --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/model.ckpt \
      --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
      --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
      --max_number_of_steps=1000 \
      --batch_size=32 \
      --learning_rate=0.01 \
      --learning_rate_decay_type=fixed \
      --save_interval_secs=60 \
      --save_summaries_secs=60 \
      --log_every_n_steps=100 \
      --optimizer=rmsprop \
      --weight_decay=0.00004
fi

if [ "$EVAL" = true ]; then
    # Run evaluation.
    python ${SLIM_DIR}eval_image_classifier.py \
      --checkpoint_path=${TRAIN_DIR} \
      --dataset_name=${DATASET_NAME} \
      --dataset_split_name=validation \
      --dataset_dir=${DATASET_DIR} \
      --model_name=${MODEL_NAME} \
      --alsologtostderr
fi

if [ "$TRAIN" = true ]; then
# Fine-tune all the new layers for 500 steps.
python ${SLIM_DIR}train_image_classifier.py \
  --train_dir=${TRAIN_DIR}/all \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME} \
  --checkpoint_path=${TRAIN_DIR} \
  --max_number_of_steps=500 \
  --batch_size=32 \
  --learning_rate=0.0001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=60 \
  --save_summaries_secs=60 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
fi

if [ "$EVAL" = true ]; then
# Run evaluation.
python ${SLIM_DIR}eval_image_classifier.py \
  --checkpoint_path=${TRAIN_DIR}/all \
  --eval_dir=${TRAIN_DIR}/all \
  --dataset_name=${DATASET_NAME} \
  --dataset_split_name=validation \
  --dataset_dir=${DATASET_DIR} \
  --model_name=${MODEL_NAME}
fi

if [ "$EXPORT" = true ]; then
    python ${SLIM_DIR}export_inference_graph.py \
      --alsologtostderr \
      --model_name=${MODEL_NAME} \
      --output_file=${MODEL_NAME}_inf_graph.pb
fi