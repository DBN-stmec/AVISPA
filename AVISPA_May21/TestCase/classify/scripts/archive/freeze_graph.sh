#!/bin/bash
TENSORFLOW_DIR=/home/jonas/code/tensorflow
CLASSIFY_DIR=../
OUTPUT_NODE=InceptionV3/Predictions/Reshape_1
$TENSORFLOW_DIR/bazel-bin/tensorflow/python/tools/freeze_graph \
  --input_graph=${CLASSIFY_DIR}models/tensorflow/inception_v3/inference_graph.pb \
  --input_checkpoint=${CLASSIFY_DIR}train/inception_v3/model.ckpt-348 \
  --input_binary=true \
  --output_graph=${CLASSIFY_DIR}models/tensorflow/inception_v3/frozen_inference_graph.pb \
  --output_node_names=${OUTPUT_NODE}