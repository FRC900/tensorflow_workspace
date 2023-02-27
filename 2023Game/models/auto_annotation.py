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
from pascal import PascalVOC, PascalObject, BndBox, size_block

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def box_to_rect(box, image_shape):
  x_min = int(min(box[1], box[3]) * image_shape[1])
  y_min = int(min(box[0], box[2]) * image_shape[0])
  x_max = int(max(box[1], box[3]) * image_shape[1])
  y_max = int(max(box[0], box[2]) * image_shape[0])
  return [x_min, y_min, x_max, y_max]

def check_iou(detected_rect, previous_labels, threshold):
  #print(f"check_iou : {detected_rect}")
  for label in previous_labels:
    new_rect = []
    new_rect.append(min(label.bndbox.xmin, label.bndbox.xmax))
    new_rect.append(min(label.bndbox.ymin, label.bndbox.ymax))
    new_rect.append(max(label.bndbox.xmin, label.bndbox.xmax))
    new_rect.append(max(label.bndbox.ymin, label.bndbox.ymax))
    #print(f"\tnew_rect = {new_rect}")
    if (bb_intersection_over_union(detected_rect, new_rect) > threshold):
      return False
  return True

# From https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou
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
    MODEL_NAME = '/home/ubuntu/tensorflow_workspace/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    # This shouldn't need to change
    PATH_TO_FROZEN_GRAPH = os.path.join(MODEL_NAME, '2023_ssd_mobilenet_v2.pb')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('/home/ubuntu/tensorflow_workspace/2023Game/data', '2023Game_label_map.pbtxt')
    PATH_TO_LABELS_NEW = os.path.join('/home/ubuntu/tensorflow_workspace/2023Game/data', '2023Game_label_map.pbtxt')

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
    category_reverse_dict = {}
    for k in category_index.keys():
        category_dict[k] = category_index[k]['name']
        category_reverse_dict[category_index[k]['name']] = k
    vis = BBoxVisualization(category_dict)
    category_index_new = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS_NEW, use_display_name=True)
    category_dict_new = {0: 'background'}
    category_reverse_dict_new = {}
    for k in category_index_new.keys():
        category_dict_new[k] = category_index_new[k]['name']
        category_reverse_dict_new[category_index_new[k]['name']] = k
    #print(category_reverse_dict_new)

    valid_labels = {}
    valid_label_names = set()
    for k in category_reverse_dict.keys():
        if k in category_reverse_dict_new:
            valid_labels[category_reverse_dict[k]] = k
            valid_label_names.add(k)

    #print(valid_labels)

    # Pick an input video to run here
    #PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/tensorflow_workspace/2023Game/data/videos'
    ########################################
    #### Code for testing against a list of images
    ####    Useful for looking at results in more detail
    ########################################
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Peak_Performance_2019_Quarterfinal_4-1.mp4_04290.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Week_2_FRC_Clips_of_the_Week_2019.mp4_01539.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Pearadox_360_Video.mp4_02940.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'FRC_Team_195_Destination_Deep_Space_in-match_Robot_Cameras.mp4_03496.png') ]
    #TEST_IMAGE_PATHS = sorted(glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'hard_neg*.png')))
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Peak_Performance_2019_Quarterfinal_4-1.mp4_04290.png') ]
    #TEST_IMAGE_PATHS = ['/home/ubuntu/tensorflow_workspace/2023Game/data/combined_88_test/untitled-f000413.png']
    TEST_IMAGE_PATHS = sorted(glob.glob(sys.argv[1]))
    print(f"TEST_IMAGE_PATHS = {TEST_IMAGE_PATHS}")
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
      #print (output_dict)
      num_detections = output_dict['num_detections']
      '''
      vis.draw_bboxes(image,
              output_dict['detection_boxes'][:num_detections],
              output_dict['detection_scores'][:num_detections],
              output_dict['detection_classes'][:num_detections],
              0.25)
      cv2.imshow(image_path, image)
      '''

    
      print(image_path)
      xml_path = image_path.rsplit('.', 1)[0] + '.xml'
      print(f"XML_PATH = {xml_path}");
 
      try:
        voc = PascalVOC.from_xml(xml_path)
      except:
        voc = PascalVOC(xml_path, size=size_block(image.shape[1], image.shape[0], 3), objects=[])

      # previous_labels is a map of obj.name -> list of previous labels of that type
      # read from the existing xml.
      # This is used to check against duplicating new labels on top of existing ones
      previous_labels = {}
      for label in valid_label_names:
          previous_labels[label] = []
      print(f"valid label names = {valid_label_names}")
      for obj in voc.objects:
         if obj.name not in valid_label_names:
            continue
         previous_labels[obj.name].append(obj)

      print(previous_labels)

      for box, sc, cl in zip(output_dict['detection_boxes'], output_dict['detection_scores'], output_dict['detection_classes']):
          if sc < 0.2:
              continue
          if cl not in valid_labels:
              continue

          label = valid_labels[cl]
          #print(f"label = {label}")

          rect = box_to_rect(box, image.shape)
          if not check_iou(rect, previous_labels[label], 0.1):
              print(f"label {label} failed IoU check")
              continue

          print(f"Adding new {label} at {rect}")
          voc.objects.append(PascalObject(label, "Unspecified", truncated=False, difficult=False, bndbox=BndBox(rect[0], rect[1], rect[2], rect[3])))

      voc.save(xml_path);
      '''
      cv2.waitKey(0) & 0xFF
      cv2.destroyWindow(image_path)
      '''

if __name__ == '__main__':
    main()

