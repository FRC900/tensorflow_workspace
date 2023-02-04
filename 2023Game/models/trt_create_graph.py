'''
Script to generate TRT output graph files from a given checkpoint

TensorRT generates optimized code for running networks on GPUs, both for
desktop and especially for the Jetson.  The optimization process takes a bit
of time, so it is possible to preprocess and save the output. 
That's the goal of this script.

Edit the SAVED_MODEL_DIR and CHECKPOINT_NUMBER variables below and run it.
'''
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from object_detection.protos import pipeline_pb2
from object_detection import exporter
from google.protobuf import text_format
import os
import subprocess

from graph_utils import force_nms_cpu as f_force_nms_cpu
from graph_utils import replace_relu6 as f_replace_relu6
from graph_utils import remove_assert as f_remove_assert

# Intermediate, unoptimized frozen graph name - make a command-line arg?
FROZEN_GRAPH_NAME='ssd_mobilenet_v2.pb'

# Output file name - make command-line arg
TRT_OUTPUT_GRAPH = 'trt_' + FROZEN_GRAPH_NAME

# Dir where model.ckpt* files are being generated - make command line arg
SAVED_MODEL_DIR='/home/ubuntu/tensorflow_workspace/2023Game/models/2023_train'
MODEL_CHECKPOINT_PREFIX='model.ckpt-' # This should be constant, no need for command line arg
CHECKPOINT_NUMBER='200000' # Make a command line arg

# Network config - make a command line arg
CONFIG_FILE=os.path.join(SAVED_MODEL_DIR, 'ssd_mobilenet_v2_512x512_coco.config')

# Graph node names for inputs and outputs - don't change unless the model graph changes
INPUT_NAME='image_tensor'
BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
NUM_DETECTIONS_NAME='num_detections'

# from tf_trt models dir
def build_detection_graph(config, checkpoint,
        batch_size=1,
        score_threshold=None,
        force_nms_cpu=True,
        replace_relu6=True,
        remove_assert=True,
        input_shape=None,
        output_dir='.generated_model'):

    """Builds a frozen graph for a pre-trained object detection model"""
    config_path = config
    checkpoint_path = checkpoint

    # parse config from file
    config = pipeline_pb2.TrainEvalPipelineConfig()
    with open(config_path, 'r') as f:
        text_format.Merge(f.read(), config, allow_unknown_extension=True)

    # override some config parameters
    if config.model.HasField('ssd'):
        config.model.ssd.feature_extractor.override_base_feature_extractor_hyperparams = True
        if score_threshold is not None:
            config.model.ssd.post_processing.batch_non_max_suppression.score_threshold = score_threshold    
        if input_shape is not None:
            config.model.ssd.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.ssd.image_resizer.fixed_shape_resizer.width = input_shape[1]
        # Docs claim enabling this might speed things up, but it seems to slow them down with TRT?
        #config.model.ssd.post_processing.batch_non_max_suppression.use_combined_nms = True
        #config.model.ssd.post_processing.batch_non_max_suppression.change_coordinate_frame = False 
    elif config.model.HasField('faster_rcnn'):
        if score_threshold is not None:
            config.model.faster_rcnn.second_stage_post_processing.score_threshold = score_threshold
        if input_shape is not None:
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.height = input_shape[0]
            config.model.faster_rcnn.image_resizer.fixed_shape_resizer.width = input_shape[1]

    if os.path.isdir(output_dir):
        subprocess.call(['rm', '-rf', output_dir])

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # export inference graph to file (initial)
    with tf.Session(config=tf_config) as tf_sess:
        with tf.Graph().as_default() as tf_graph:
            exporter.export_inference_graph(
                'image_tensor', 
                config, 
                checkpoint_path, 
                output_dir, 
                input_shape=[batch_size, None, None, 3]
            )

    # remove temporary directory after saving frozen graph output
    os.rename(os.path.join(output_dir, 'frozen_inference_graph.pb'), os.path.join(output_dir, FROZEN_GRAPH_NAME))
    os.rename(os.path.join(output_dir, FROZEN_GRAPH_NAME), os.path.join(SAVED_MODEL_DIR, FROZEN_GRAPH_NAME))
    subprocess.call(['rm', '-rf', output_dir])

    # read frozen graph from file
    frozen_graph = tf.GraphDef()
    with tf.io.gfile.GFile(os.path.join(SAVED_MODEL_DIR, FROZEN_GRAPH_NAME), 'rb') as f:
        frozen_graph.ParseFromString(f.read())
    
    # apply graph modifications - fix stuff to make TensorRT optimizations faster
    if force_nms_cpu:
        frozen_graph = f_force_nms_cpu(frozen_graph)
    if replace_relu6:
        frozen_graph = f_replace_relu6(frozen_graph)
    if remove_assert:
        frozen_graph = f_remove_assert(frozen_graph)

    # get input and output tensor names
    # TODO: handle mask_rcnn 
    input_names = [INPUT_NAME]
    output_names = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

    return frozen_graph, input_names, output_names


def main():
    # What model to run from - should be the directory name of an exported trained model
    # Change me to the directory checkpoint files are saved in
    # Note - runing score_threshold here is a good idea. It drops detections below that confidence score,
    # speeding up the inference. This should probably be tuned in conjunction with the confidence threshold
    # in higher level obj detect code - no point in keeping objects which will never be used by that code
    frozen_graph, input_names, output_names = build_detection_graph(
        config=CONFIG_FILE,
        checkpoint=os.path.join(SAVED_MODEL_DIR, MODEL_CHECKPOINT_PREFIX+CHECKPOINT_NUMBER),
        score_threshold=0.2,
        batch_size=1
    )
    '''
    trt_graph = trt.create_inference_graph(
        input_graph_def=frozen_graph,
        outputs=output_names,
        max_batch_size=1,
        max_workspace_size_bytes=1 << 25,
        precision_mode='FP32', # TODO - FP16 or INT8 for Jetson
        minimum_segment_size=50
    )
    '''
    trt_graph = frozen_graph

    with tf.io.gfile.GFile(os.path.join(SAVED_MODEL_DIR, TRT_OUTPUT_GRAPH), 'wb') as f:
        f.write(trt_graph.SerializeToString())

if __name__ == '__main__':
    main()
