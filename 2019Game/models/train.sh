#!/bin/bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.1/compat/lib.real
PIPELINE_CONFIG_PATH=/home/ubuntu/tensorflow_workspace/2019Game/models/model/ssdlite_mobilenet_v2_coco.config
MODEL_DIR=/home/ubuntu/tensorflow_workspace/2019Game/models
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python /home/ubuntu/models/research/object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr



