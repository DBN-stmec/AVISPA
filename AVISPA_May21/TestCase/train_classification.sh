#!/usr/bin/env bash
MODEL="vgg16"
DATASET="/home/duybao/PTW_AVISPA/AVISPA_May21/TestCase/data/RNGN19co/"
EPOCHS=1
MODEL_NAME="vgg16_Model"
#python3 train.py --model=$MODEL --data_dir=$DATASET --nb_epoch=$EPOCHS --model_name=$MODEL_NAME |tee /home/a.ziegenbein/TestCase/out/out_vgg16_kaggle_train.txt
python3 train.py --model=$MODEL --data_dir=$DATASET --nb_epoch=$EPOCHS --model_name=$MODEL_NAME |tee /home/duybao/PTW_AVISPA/AVISPA_May21/TestCase/Test_Case_output.txt
