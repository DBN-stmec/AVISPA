#!/usr/bin/env bash
DATA_DIR=$1
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
EXPORT_DIR="$SCRIPT_DIR/../export"
TRAIN_DIR="$DATA_DIR/train"
VALIDATION_DIR="$DATA_DIR/validation"
REST_DIR="$DATA_DIR/rest"
ALL_DIR="$DATA_DIR/all"
SPLIT=80
declare -a LABELS=("ok" "verschlissen")
echo $EXPORT_DIR
if [ ! -d "$DATA_DIR" ]; then
  mkdir "$DATA_DIR"
fi
if [ ! -d "$TRAIN_DIR" ]; then
  mkdir "$TRAIN_DIR"
fi
if [ ! -d "$VALIDATION_DIR" ]; then
  mkdir "$VALIDATION_DIR"
fi
if [ ! -d "$ALL_DIR" ]; then
  mkdir "$ALL_DIR"
fi
if [ ! -d "$REST_DIR" ]; then
  mkdir "$REST_DIR"
fi
for LABEL in "${LABELS[@]}"
do
    echo "Copying files from $EXPORT_DIR/$LABEL to $ALL_DIR"
    cp -R "$EXPORT_DIR/$LABEL" "$ALL_DIR"
    cp -R "$EXPORT_DIR/$LABEL" "$REST_DIR"
    echo "Creating directory $TRAIN_DIR/$LABEL"
    mkdir "$TRAIN_DIR/$LABEL"
    echo "Creating directory $VALIDATION_DIR/$LABEL"
    mkdir "$VALIDATION_DIR/$LABEL"
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