#!/bin/bash
# Need pretrained model info for transfer learning setup
# if [ ! -f ssd_mobilenet_v2_coco_2018_03_29/checkpoint ]
# cd /home/ubuntu/tensorflow_workspace/2020Game/models
# wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# tar -xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# fi
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib/stubs
MODEL_DIR=/home/ubuntu/tensorflow_workspace/2020Game/models/INSERT_TRAINING_DIR_HERE
PIPELINE_CONFIG_PATH=$MODEL_DIR/ssd_mobilenet_v2_coco.config
NUM_TRAIN_STEPS=500000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python3 /home/ubuntu/tensorflow_workspace/2020Game/models/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
