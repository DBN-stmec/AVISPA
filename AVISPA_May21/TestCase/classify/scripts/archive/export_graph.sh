#!/bin/bash
TENSORFLOW_DIR=/home/jonas/code/tensorflow
CLASSIFY_DIR=../
OUTPUT_NODE=InceptionV3/Predictions/Reshape_1
MODEL_NAME=inception_v3
$TENSORFLOW_DIR/bazel-bin/tensorflow/python/tools/freeze_graph \
  --alsologtostderr \
  --model_name=${MODEL_NAME} \
  --output_file=${CLASSIFY_DIR}models/tensorflow/${MODEL_NAME}/inference_graph.pb