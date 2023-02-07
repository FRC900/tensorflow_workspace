"""
Run object detection a video or set of images
"""
import cv2
import numpy as np
import sys
import tensorflow as tf
import os
import glob
import timing
from visualization import BBoxVisualization




# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util

if len(sys.argv) > 1:
  video_name = sys.argv[1]
else:
  print('No video parameter passed.')


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
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
      output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def main():
    # What model to run from - should be the directory name of an exported trained model
    # Change me to the directory exported using the export_inference_graph.py command
    MODEL_NAME = '/home/ubuntu/tensorflow_workspace/2023Game/models/2023_train'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    # This shouldn't need to change
    PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME,'ssd_mobilenet_v2.pb')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('/home/ubuntu/tensorflow_workspace/2023Game/data', '2023Game_label_map.pbtxt')

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
    category_dict = {0: 'background'}
    for k in category_index.keys():
        category_dict[k] = category_index[k]['name']
    vis = BBoxVisualization(category_dict)

    # Pick an input video to run here
    PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/tensorflow_workspace/2023Game/data/videos'
    if len(sys.argv) > 1:
      cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, video_name))
    else:
      cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2023 Field Tour Video_ Substations.mp4'))
    # Used to write annotated video (video with bounding boxes and labels) to an output mp4 file
    #vid_writer = cv2.VideoWriter(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2020_INFINITE_RECHARGE_Field_Drone_Video_Field_from_Alliance_Station_annotated.mp4'), cv2.VideoWriter_fourcc(*"FMP4"), 30., (1920,1080))

    display_viz = True # Make command line arg
    t = timing.Timings()

    while(True):
      t.start('frame')
      t.start('vid')
      ret, cv_vid_image = cap.read()
      cv_vid_image = cv2.pyrDown(cv_vid_image)
      t.end('vid')
      if not ret:
        break

      next_frame = False
      while (not next_frame):
        # Vid input is BGR, need to convert to RGB and resize 
        # to net input size to run inference
        t.start('cv')
        image_resized = cv2.resize(cv_vid_image, (300,300))
        image_np = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        # Expand dimensions since the model expects images to have shape: [batch_size = 1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        t.end('cv')

        # Actual detection.
        t.start('inference')
        output_dict = run_inference_for_single_image(image_np_expanded, sess, detection_graph)
        t.end('inference')

        if display_viz:
          t.start('viz')
          # output_dictionary will have detection box coordinates, along with the classes
          # (index of the text labels) and confidence scores for each detection
          print(output_dict)
          num_detections = output_dict['num_detections']
          vis.draw_bboxes(cv_vid_image,
                  output_dict['detection_boxes'][:num_detections],
                  output_dict['detection_scores'][:num_detections],
                  output_dict['detection_classes'][:num_detections],
                  0.25)
          '''
          Much slower version using tf vis_util
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
              min_score_thresh=0.25,
              groundtruth_box_visualization_color='yellow')
          '''
          cv2.imshow('img', cv_vid_image)
          #vid_writer.write(cv_vid_image)
          t.end('viz')
          key = cv2.waitKey(1) & 0xFF
          if key == 27:
             return
        next_frame = True
        t.end('frame')


    """
    ########################################
    #### Code for testing against a list of images
    ####    Useful for looking at results in more detail
    ########################################
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Peak_Performance_2019_Quarterfinal_4-1.mp4_04290.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Week_2_FRC_Clips_of_the_Week_2019.mp4_01539.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Pearadox_360_Video.mp4_02940.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'FRC_Team_195_Destination_Deep_Space_in-match_Robot_Cameras.mp4_03496.png') ]
    TEST_IMAGE_PATHS = sorted(glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'hard_neg*.png')))
    for image_path in TEST_IMAGE_PATHS:
      image = cv2.imread(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      #image_np = load_image_into_numpy_array(image)
      image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      #image_np = cv2.pyrDown(cv2.pyrDown(image_np));
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
          min_score_thresh=0.20)
      cv2.imshow(image_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
      cv2.waitKey(0) & 0xFF
      cv2.destroyWindow(image_path)
    """

if __name__ == '__main__':
    main()

