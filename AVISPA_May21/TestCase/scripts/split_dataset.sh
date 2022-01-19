#!/usr/bin/env bash

# Splits this structure:
# - all
# -- ok [100]
# -- verschlissen [100]
# into this:
# - all
# -- ok [100]
# -- verschlissen [100]
# - train
# -- ok [80]
# -- verschlissen [80]
# - validation
# -- ok [20]
# -- verschlissen [20]
# Leftover files will be put in
# - rest
# A directory consisting of train + validation images with even distribution will be created:
# - even

ALL_DIR=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REST_DIR="$ALL_DIR/../rest"
EVEN_DIR="$ALL_DIR/../even"
TRAIN_DIR="$ALL_DIR/../train"
VALIDATION_DIR="$ALL_DIR/../validation"
SPLIT=80
declare -a LABELS=("ok" "verschlissen")
echo $ALL_DIR
mkdir -p "$REST_DIR"
mkdir -p "$EVEN_DIR"
mkdir -p "$TRAIN_DIR"
mkdir -p "$VALIDATION_DIR"
for LABEL in "${LABELS[@]}"
do
    echo "Creating directory $REST_DIR/$LABEL"
    echo "Copying files from $ALL_DIR/$LABEL to $REST_DIR/$LABEL"
    cp -R "$ALL_DIR/$LABEL" "$REST_DIR"
    echo "Creating directory $TRAIN_DIR/$LABEL"
    mkdir -p "$TRAIN_DIR/$LABEL"
    echo "Creating directory $VALIDATION_DIR/$LABEL"
    mkdir -p "$VALIDATION_DIR/$LABEL"
done

MIN_COUNT=9999999
for LABEL in "${LABELS[@]}"
do
    cd "$REST_DIR/$LABEL"
    shopt -s dotglob nullglob
    a=(*)
    COUNT=${#a[@]}
    echo "Found $COUNT files with label '$LABEL'"
    if [ "$COUNT" -lt "$MIN_COUNT" ]; then
        MIN_COUNT=$COUNT
    fi
done
TRAIN_COUNT=$(( SPLIT * MIN_COUNT / 100 ))
VALIDATION_COUNT="$(( $MIN_COUNT - $TRAIN_COUNT ))"
echo "Dataset is randomly split into $TRAIN_COUNT train and $VALIDATION_COUNT validation images"
for LABEL in "${LABELS[@]}"
do
    cd "$REST_DIR/$LABEL"
    shuf -n $TRAIN_COUNT -e * | xargs -i mv {} $TRAIN_DIR/$LABEL/
    shuf -n $VALIDATION_COUNT -e * | xargs -i mv {} $VALIDATION_DIR/$LABEL/
done
for LABEL in "${LABELS[@]}"
do
    cp -R "$TRAIN_DIR/$LABEL" "$EVEN_DIR/"
    cp -R "$VALIDATION_DIR/$LABEL" "$EVEN_DIR/"
done