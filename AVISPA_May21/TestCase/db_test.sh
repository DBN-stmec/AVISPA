#!/usr/bin/env bash
MODEL="xception"
DATASET="/home/duybao/PTW_AVISPA/AVISPA_May21/TestCase/data/RNGN19co/"
#EPOCHS=50
EPOCHS=5 #Testweise 5 Epochen, eigentlich sollen es 50 Epochen sein
# Run the trained model(name) in this folder: AVISPA/TestCase/classify/trained/
MODEL_NAME="xception_RNGN19co"
#python3 train.py --model=$MODEL --data_dir=$DATASET --nb_epoch=$EPOCHS --model_name=$MODEL_NAME |tee /home/duybao/PTW_AVISPA/AVISPA_May21/TestCase/out/out_vgg16_RNGN19co_train.txt
python3 process.py --source="/home/duybao/PTW_AVISPA/AVISPA_May21/TestCase/data/single_img" --skip_detection --classification_model=$MODEL --classification_model_name=$MODEL_NAME --log_level="debug" |tee /home/duybao/PTW_AVISPA/AVISPA_May21/TestCase/out/out_xception_RNGN19co_pred.txt