#!/bin/bash
# Need pretrained model info for transfer learning setup
# if [ ! -f ssd_mobilenet_v2_coco_2018_03_29/checkpoint ]
# cd /home/ubuntu/tensorflow_workspace/2022Game/models
# wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# tar -xzf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# rm ssd_mobilenet_v2_coco_2018_03_29.tar.gz
# fi
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib/stubs
MODEL_DIR=/home/ubuntu/tensorflow_workspace/2022Game/models/2022_v3
PIPELINE_CONFIG_PATH=$MODEL_DIR/ssd_mobilenet_v2_512x512_coco.config
NUM_TRAIN_STEPS=200000
SAMPLE_1_OF_N_EVAL_EXAMPLES=100
python3 /home/ubuntu/tensorflow_workspace/2022Game/models/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr
