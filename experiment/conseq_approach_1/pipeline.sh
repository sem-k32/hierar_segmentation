#!/bin/bash
# remove previous results
if [ -d "./results" ];
then
echo "Removing previous results"
rm -r ./results
fi
# preprocessing stage
echo "Preprocessing stage"
python preprocessing_stage.py
# train model 1 stage
echo "Train model 1 stage"
cd model_1_stage
python train_model_1_stage.py
cd ..
# train model 2 stage
echo "Train model 2 stage"
cd model_2_stage
python train_model_2_stage.py
cd ..
# train model 3 stage
echo "Train model 3 stage"
cd model_3_stage
python train_model_3_stage.py
cd ..
# train model 4 stage
echo "Train model 4 stage"
cd model_4_stage
python train_model_4_stage.py
cd ..
# build final model, calculate metrics, vizualize segmentations
echo "Final model stage"
cd final_model_stage
python python valid_segmentation.py 
cd ..