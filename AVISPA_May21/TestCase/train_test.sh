#!/usr/bin/env bash
MODEL="xception"
DATASET="/home/a.ziegenbein/TestCase/data/RNGN19co"
EPOCHS=500
MODEL_NAME="xception_RNGN19co"
python3 train.py --model=$MODEL --data_dir=$DATASET --nb_epoch=$EPOCHS --model_name=$MODEL_NAME |tee /home/a.ziegenbein/TestCase/out/out_xception_RNGN19co_train.txt
python3 process.py --source='/home/a.ziegenbein/TestCase/data/RNGN19co/test' --skip_detection --classification_model=xception --classification_model_name=xception_RNGN19co --log_level="debug" |tee /home/a.ziegenbein/TestCase/out/out_xception_RNGN19co_pred.txt
MODEL="xception"
DATASET="/home/a.ziegenbein/TestCase/data/RNGN19cross"
EPOCHS=500
MODEL_NAME="xception_RNGN19cross"
python3 train.py --model=$MODEL --data_dir=$DATASET --nb_epoch=$EPOCHS --model_name=$MODEL_NAME |tee /home/a.ziegenbein/TestCase/out/out_xception_RNGN19cross_train.txt
python3 process.py --source='/home/a.ziegenbein/TestCase/data/RNGN19cross/test' --skip_detection --classification_model=xception --classification_model_name=xception_RNGN19cross --log_level="debug" |tee /home/a.ziegenbein/TestCase/out/out_xception_RNGN19cross_pred.txt
MODEL="inception_v3"
DATASET="/home/a.ziegenbein/TestCase/data/RNGN19co"
EPOCHS=500
MODEL_NAME="inception_v3_RNGN19co"
python3 train.py --model=$MODEL --data_dir=$DATASET --nb_epoch=$EPOCHS --model_name=$MODEL_NAME |tee /home/a.ziegenbein/TestCase/out/out_inception_v3_RNGN19co_train.txt
python3 process.py --source='/home/a.ziegenbein/TestCase/data/RNGN19co/test' --skip_detection --classification_model=inception_v3 --classification_model_name=inception_v3_RNGN19co --log_level="debug" |tee /home/a.ziegenbein/TestCase/out/out_inception_v3_RNGN19co_pred.txt
MODEL="inception_v3"
DATASET="/home/a.ziegenbein/TestCase/data/RNGN19cross"
EPOCHS=500
MODEL_NAME="inception_v3_RNGN19cross"
python3 train.py --model=$MODEL --data_dir=$DATASET --nb_epoch=$EPOCHS --model_name=$MODEL_NAME |tee /home/a.ziegenbein/TestCase/out/out_inception_v3_RNGN19cross_train.txt
python3 process.py --source='/home/a.ziegenbein/TestCase/data/RNGN19cross/test' --skip_detection --classification_model=inception_v3 --classification_model_name=inception_v3_RNGN19cross --log_level="debug" |tee /home/a.ziegenbein/TestCase/out/out_inception_v3_RNGN19cross_pred.txt
MODEL="vgg16"
DATASET="/home/a.ziegenbein/TestCase/data/RNGN19co"
EPOCHS=500
MODEL_NAME="vgg16_RNGN19co"
python3 train.py --model=$MODEL --data_dir=$DATASET --nb_epoch=$EPOCHS --model_name=$MODEL_NAME |tee /home/a.ziegenbein/TestCase/out/out_vgg16_RNGN19co_train.txt
python3 process.py --source='/home/a.ziegenbein/TestCase/data/RNGN19co/test' --skip_detection --classification_model=vgg16 --classification_model_name=vgg16_RNGN19co --log_level="debug" |tee /home/a.ziegenbein/TestCase/out/out_vgg16_RNGN19co_pred.txt
MODEL="vgg16"
DATASET="/home/a.ziegenbein/TestCase/data/RNGN19cross"
EPOCHS=500
MODEL_NAME="vgg16_RNGN19cross"
python3 train.py --model=$MODEL --data_dir=$DATASET --nb_epoch=$EPOCHS --model_name=$MODEL_NAME |tee /home/a.ziegenbein/TestCase/out/out_vgg16_RNGN19cross_train.txt
python3 process.py --source='/home/a.ziegenbein/TestCase/data/RNGN19cross/test' --skip_detection --classification_model=vgg16 --classification_model_name=vgg16_RNGN19cross --log_level="debug" |tee /home/a.ziegenbein/TestCase/out/out_vgg16_RNGN19cross_pred.txt