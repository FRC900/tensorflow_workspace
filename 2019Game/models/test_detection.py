
import cv2
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
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
    # What model to download.
    MODEL_NAME = '/home/ubuntu/tensorflow_workspace/2019Game/models/exported_graphs_cocov2_decaysteps800720'
    #MODEL_NAME = '/home/ubuntu/tensorflow_workspace/2019Game/models/ssd_mobilenet_v2_coco_2018_03_29'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('/home/ubuntu/tensorflow_workspace/2019Game/data', '2019Game_label_map.pbtxt')
    #PATH_TO_LABELS = '/home/ubuntu/models/research/object_detection/data/mscoco_label_map.pbtxt'
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

    PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/tensorflow_workspace/2019Game/data/videos'
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Peak_Performance_2019_Quarterfinal_4-1.mp4_04290.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Week_2_FRC_Clips_of_the_Week_2019.mp4_01539.png') ]
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Pearadox_360_Video.mp4_02940.png') ]
    TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'FRC_Team_195_Destination_Deep_Space_in-match_Robot_Cameras.mp4_03496.png') ]

    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)

    #cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Peak_Performance_2019_Quarterfinal_4-1.mp4'))
    cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Pearadox_360_Video.mp4'))
    vid_writer = cv2.VideoWriter(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Pearadox_360_Video_annotated.avi'), cv2.VideoWriter_fourcc(*"FMP4"), 30., (640,360))
    #for image_path in TEST_IMAGE_PATHS:
    while(True):
      ret, cv_vid_image = cap.read()
      next_frame = False
      while (not next_frame):
        image_np = cv2.cvtColor(cv_vid_image, cv2.COLOR_BGR2RGB)
        #image = Image.open(image_path)
        #resized_image = image.resize((256,256), Image.ANTIALIAS)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        #image_np = load_image_into_numpy_array(image)
        #resized_image_np = load_image_into_numpy_array(resized_image)
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
            line_thickness=8)
        cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        vid_writer.write(cv2.pyrDown(cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)))
        next_frame = True
        key = cv2.waitKey(10) & 0xFF
        if key == ord("f"):
          next_frame = True
        #plt.figure(figsize=IMAGE_SIZE)
        #plt.imshow(image_np)
        #plt.pause(20)

if __name__ == '__main__':
    main()

'''
import numpy as np
import os
import tensorflow as tf
import cv2
#from picamera.array import PiRGBArray
#from picamera import PiCamera
# Import utilities
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# Define import paths
PATH_TO_CKPT = 'output_inference_graph-1.4.1.pb/frozen_inference_graph.pb' # Import frozen model
PATH_TO_LABELS = 'frc_label_map.pbtxt' # Import map of labels
NUM_CLASSES=1 # Only one class (box)
# Load frozen model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
# Load Label Map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# Helper function for data format
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
# Define function for detecting objects within an image
def detect_objects(image_np, sess, detection_graph):
  #Define input
  image_np_expanded = np.expand_dims(image_np, axis=0)
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  
  #Define outputs
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  
  #Predict
  (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_np_expanded})
  
  #Visualize
  vis_util.visualize_boxes_and_labels_on_image_array(
  image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
  np.squeeze(scores), category_index, use_normalized_coordinates=True,
  min_score_thresh=.15,
  line_thickness=4)
  
  return image_np
  
# Define function for handling images
def detect_image(image_path, sess, detection_graph):
  #Import image
  image = cv2.imread(image_path)
  image = cv2.resize(image, (480, 640))
  image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  #Detect objects
  image_np = detect_objects(image_np, sess, detection_graph)
  #cv2.imwrite(output, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  cv2.waitKey(0)
def detect_image_webcam(image, sess, detection_graph):
  # Format data
  #image = cv2.resize(image, (480, 640))
  image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  # Detect objects
  image_np = detect_objects(image_np, sess, detection_graph)
  #cv2.imwrite(output, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  cv2.waitKey(10)
  return image_np
def detect_objects_coords(image_np, sess, detection_graph):
    # Define input
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Define outputs
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    # Predict
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Find box vertices
    box_coords = []
    for i in range(0, len(scores[0])):
        if scores[0][0] <= .84:
            return "Nothing here"
        if scores[0][i] > .84:
            box = boxes[0][i]
            rows = image_np.shape[0]
            cols = image_np.shape[1]
            box[0] = box[0]*rows
            box[1] = box[1]*cols
            box[2] = box[2]*rows
            box[3] = box[3]*cols
            box_coords.append(box)
            #cv2.rectangle(image_np, (box[1], box[0]), (box[3], box[2]), (0,255,0),3)
        else:
            break
    #cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    #cv2.waitKey(1)
    
    # Returns coords of box in [y1, x1, y2, x2] format
    return box_coords
    
def detect_image_coords(image_path, sess, detection_graph):
            #Import image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (480, 640))
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Detect objects
            coords = detect_objects_coords(image_np, sess, detection_graph)
            return coords
import sys
# Detect images
if (__name__ == '__main__'):
    # For reading from image files
    #input_dir = '../frcbox_test_video/images/'
    #input_dir = '../test_images/'
    #image_paths = sorted([ input_dir + f for f in os.listdir(input_dir)])
    #inp = sys.argv[1]
    # Start Session
    with detection_graph.as_default():
        sess = tf.Session(graph=detection_graph)
    
    # Test Coord output
    #print(detect_image_coords(image_paths[5], sess, detection_graph))
    #detect_image_coords(inp, sess, detection_graph)
    # Loop through images and detect boxes (for image files)
    #for i in range(0, len(image_paths)):
    #   detect_image(image_paths[i], sess, detection_graph)
    
    # Loop through webcam frames
    #cam = PiCamera()
    #cap = PiRGBArray(cam)
    cap = cv2.VideoCapture("/home/kjaget/Downloads/2018_Field_Video_Scale_360p.mp4")
    while(True):
        #cam.capture(cap, format="bgr")
        #image= cap.array
        ret,image = cap.read()
        # Pad to 4:3 aspect ratio
        image = cv2.copyMakeBorder(image, 60, 60, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        image = cv2.pyrUp(image)
        
        detect_image_webcam(image, sess, detection_graph)
        #image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Detect objects
        #coords = detect_objects_coords(image_np, sess, detection_graph)
        #print(coords)
        #cv2.imshow('img',image)
        #key = cv2.waitKey(1) & 0xFF
        #cap.truncate(0)
        #if key == ord("q"):
        #    break
    #frame_np = detect_image_webcam(frame, sess, detection_graph)
    #   
    #   if cv2.waitKey(1) & 0xFF == ord('q'):
    #       break
        
'''
