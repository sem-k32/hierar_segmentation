#!/bin/bash
# remove previous results
if [ -d "./results" ];
then
rm -r ./results
fi
# preprocessing stage
echo "Preprocessing stage"
python preprocessing_stage.py
# train stage
echo "Train stage"
python train_stage.py