"""

Run detection using optimized trt GPU code.

"""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
import cv2
import os
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util

def get_frozen_graph(pb_path):
    """Read Frozen Graph file from disk."""
    """
    with tf.gfile.FastGFile(pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def
    """
    """Load the TRT graph from the pre-build pb file."""
    trt_graph_def = tf.GraphDef()
    with tf.gfile.GFile(pb_path, 'rb') as pf:
        trt_graph_def.ParseFromString(pf.read())

    """
    # force CPU device placement for NMS ops
    for node in trt_graph_def.node:
        if 'rfcn_' in pb_path and 'SecondStage' in node.name:
            node.device = '/device:GPU:0'
        if 'faster_rcnn_' in pb_path and 'SecondStage' in node.name:
            node.device = '/device:GPU:0'
        if 'NonMaxSuppression' in node.name:
            node.device = '/device:CPU:0'
    """
    with tf.Graph().as_default() as trt_graph:
        tf.import_graph_def(trt_graph_def, name='')
    return trt_graph

# Takes a image, and using the tensorflow session and graph
# provided, runs inference on the image. This returns a list
# of detections - each includes the object bounding box, type
# and confidence
def run_inference_for_single_image(image, sess):
    input_names = ['image_tensor']
    tf_input = sess.graph.get_tensor_by_name(input_names[0] + ':0')
    tf_scores = sess.graph.get_tensor_by_name('detection_scores:0')
    tf_boxes = sess.graph.get_tensor_by_name('detection_boxes:0')
    tf_classes = sess.graph.get_tensor_by_name('detection_classes:0')
    tf_num_detections = sess.graph.get_tensor_by_name('num_detections:0')
    scores, boxes, classes, num_detections = sess.run([tf_scores, tf_boxes, tf_classes, tf_num_detections], feed_dict={
        tf_input: image[None, ...]
    })
    output_dict = { }
    output_dict['detection_boxes'] = boxes[0]  # index by 0 to remove batch dimension
    output_dict['detection_scores'] = scores[0]
    output_dict['detection_classes'] = classes[0].astype(np.int64)
    output_dict['num_detections'] = int(num_detections[0]);
    return output_dict

def main():
    # Dir where model.ckpt* files are being generated
    SAVED_MODEL_DIR='/home/ubuntu/tensorflow_workspace/2020Game/models/tmp4'
    TRT_OUTPUT_GRAPH = 'trt_graph.pb'

    # The TensorRT inference graph file downloaded from Colab or your local machine.
    pb_fname = os.path.join(SAVED_MODEL_DIR, TRT_OUTPUT_GRAPH)
    trt_graph = get_frozen_graph(pb_fname)

    # Create session and load graph
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=tf_config, graph=trt_graph)

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('/home/ubuntu/tensorflow_workspace/2020Game/data', '2020Game_label_map.pbtxt')
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/tensorflow_workspace/2020Game/data/videos'

    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_Field_Tour_Video_Alliance_Station.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_Field_Tour_Video_Loading_Bay.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_Field_Tour_Video_Power_Port.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_Field_Tour_Video_Rockwell.mp4'))
    cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Footage_Control_Panel.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Cross_Field_Views.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Cross_Field_Views_1080p.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Field_from_Alliance_Station.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Field_from_Alliance_Station_1080p.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Shield_Generator.mp4'))

    # Used to write annotated video (video with bounding boxes and labels) to an output mp4 file
    #vid_writer = cv2.VideoWriter(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_Field_Tour_Video_Power_Port_annotated.avi'), cv2.VideoWriter_fourcc(*"FMP4"), 30., (640,360))
    while(True):
      ret, cv_vid_image = cap.read()
      next_frame = False
      while (not next_frame):
        # Vid input is BGR, need to convert to RGB and resize 
        # to 300x300 to run inference
        image_rgb = cv2.cvtColor(cv_vid_image, cv2.COLOR_BGR2RGB)
        image300x300 = cv2.resize(image_rgb, (300,300))
        output_dict = run_inference_for_single_image(image300x300, tf_sess)

        # output_dictionary will have detection box coordinates, along with the classes
        # (index of the text labels) and confidence scores for each detection
        print output_dict

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_rgb,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=4,
            max_boxes_to_draw=50,
            min_score_thresh=0.35,
            groundtruth_box_visualization_color='yellow')
        cv2.imshow('img', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        #vid_writer.write(cv2.pyrDown(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)))
        key = cv2.waitKey(5) & 0xFF
        if key == ord("f"):
          next_frame = True
        next_frame = True

if __name__ == '__main__':
    main()
