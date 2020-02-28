
import cv2
import glob
import numpy as np
import os
import sys
import tensorflow as tf

from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Takes a image, and using the tensorflow session and graph
# provided, runs inference on the image. This returns a list
# of detections - each includes the object bounding box, type
# and confidence
def run_inference_for_single_image(image, sess, graph):
  with graph.as_default():
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
      tensor_name = key + ':0'
      if tensor_name in all_tensor_names:
        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
            tensor_name)
    if 'detection_masks' in tensor_dict:
      # The following processing is only for single image
      detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
      detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
      # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
      real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
      detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
      detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
      detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
          detection_masks, detection_boxes, image.shape[1], image.shape[2])
      detection_masks_reframed = tf.cast(
          tf.greater(detection_masks_reframed, 0.5), tf.uint8)
      # Follow the convention by adding back the batch dimension
      tensor_dict['detection_masks'] = tf.expand_dims(
          detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: image})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
      output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def main():
    # What model to run from - should be the directory name of an exported trained model
    # Change me to the directory exported using the export_inference_graph.py command
    MODEL_NAME = '/home/ubuntu/tensorflow_workspace/2020Game/models/tmp5'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    # This shouldn't need to change
    PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME, os.path.join('frozen', 'frozen_inference_graph.pb'))

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('/home/ubuntu/tensorflow_workspace/2020Game/data', '2020Game_label_map.pbtxt')

    # Init TF detection graph and session
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        sess = tf.Session(graph=detection_graph)
    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    # Pick an input video to run here
    PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/tensorflow_workspace/2020Game/data/videos'
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_Field_Tour_Video_Alliance_Station.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_Field_Tour_Video_Loading_Bay.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_Field_Tour_Video_Power_Port.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Footage_Control_Panel.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Cross_Field_Views.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Cross_Field_Views_1080p.mp4'))
    cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Field_from_Alliance_Station.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Field_from_Alliance_Station_1080p.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Shield_Generator.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Nashua_Week0_Final_01.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Nashua_Week0_Match01.mp4'))
    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'VID_20200215_145921.mp4'))

    # Used to write annotated video (video with bounding boxes and labels) to an output mp4 file
    #vid_writer = cv2.VideoWriter(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Field_from_Alliance_Station_annotated.mp4'), cv2.VideoWriter_fourcc(*"FMP4"), 30., (1920,1080))
    while(True):
      ret, cv_vid_image = cap.read()
      next_frame = False
      while (not next_frame):
        image_np = cv2.cvtColor(cv_vid_image, cv2.COLOR_BGR2RGB)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        #resized_image_np_expanded = np.expand_dims(resized_image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, sess, detection_graph)
        # output_dictionary will have detection box coordinates, along with the classes
        # (index of the text labels) and confidence scores for each detection
        print output_dict
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
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
        cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        #vid_writer.write(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(5) & 0xFF
        if key == ord("f"):
          next_frame = True
        next_frame = True

    """
    ########################################
    #### Code for testing against a list of images
    ####    Useful for looking at results in more detail
    ########################################
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Peak_Performance_2019_Quarterfinal_4-1.mp4_04290.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Week_2_FRC_Clips_of_the_Week_2019.mp4_01539.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Pearadox_360_Video.mp4_02940.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'FRC_Team_195_Destination_Deep_Space_in-match_Robot_Cameras.mp4_03496.png') ]
    TEST_IMAGE_PATHS = sorted(glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.JPG')))
    for image_path in TEST_IMAGE_PATHS:
      image = cv2.imread(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      #image_np = load_image_into_numpy_array(image)
      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image_np = cv2.pyrDown(cv2.pyrDown(image_np));
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      #resized_image_np_expanded = np.expand_dims(resized_image_np, axis=0)
      # Actual detection.
      output_dict = run_inference_for_single_image(image_np_expanded, sess, detection_graph)
      # Visualization of the results of a detection.
      print output_dict
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          category_index,
          instance_masks=output_dict.get('detection_masks'),
          use_normalized_coordinates=True,
          line_thickness=4,
          max_boxes_to_draw=50,
          min_score_thresh=0.30)
      cv2.imshow(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
      cv2.waitKey(0) & 0xFF
      cv2.destroyWindow(image_path)
    """

if __name__ == '__main__':
    main()

