#!/bin/bash
cd ..
declare -a DATASETS=(
    "/media/jonas/data/data/classify/full/training/3"
    "/media/jonas/data/data/classify/frontal/training/1"
    "/media/jonas/data/data/classify/frontal_cropped/training/1"
    "/media/jonas/data/data/classify/frontal_grayscale_cropped/training/1"
    "/media/jonas/data/data/classify/frontal_grayscale_cropped_ahe/training/1"
    "/media/jonas/data/data/classify/frontal_cropped_cs/training/1"
    "/media/jonas/data/data/classify/frontal_grayscale_cropped_cs/training/1"
    )

EPOCHS=500

echo "Evaluation on /test"
COUNT=0
for DATASET in "${DATASETS[@]}"
do
    COUNT=$(($COUNT+1))
    echo "Evaluation on $DATASET/test"
    python3 process.py --source=$DATASET/test --skip_detection --classification_model_name="A$COUNT"
    python3 process.py --source=$DATASET/test --skip_detection --classification_model_name="B$COUNT"
    python3 process.py --source=$DATASET/test --skip_detection --classification_model_name="C$COUNT"
    python3 process.py --source=$DATASET/test --skip_detection --classification_model_name="D$COUNT"
done

paplay /usr/share/sounds/freedesktop/stereo/complete.oga