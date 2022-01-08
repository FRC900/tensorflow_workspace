# Takes a directory saved by the best_exporter code and converts it to a frozen graph
import tensorflow as tf
import os

pb_saved_model = "/home/ubuntu/tensorflow_workspace/2020Game/models/trained_ssd_mobilenet_v2_coco_focal_loss/export/best_exporter/1590476859"
BOXES_NAME='detection_boxes'
CLASSES_NAME='detection_classes'
SCORES_NAME='detection_scores'
NUM_DETECTIONS_NAME='num_detections'
OUTPUT_NAMES = [BOXES_NAME, CLASSES_NAME, SCORES_NAME, NUM_DETECTIONS_NAME]

_graph = tf.Graph()
with _graph.as_default():
    _sess = tf.Session(graph=_graph)
    model = tf.saved_model.loader.load(_sess, ["serve"], pb_saved_model)
    graphdef = tf.get_default_graph().as_graph_def()
    frozen_graph = tf.graph_util.convert_variables_to_constants(_sess,graphdef, OUTPUT_NAMES)
    frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

with tf.gfile.GFile(os.path.join(pb_saved_model, "frozen_inference_graph.pb"), "wb") as f:
    f.write(frozen_graph.SerializeToString())
