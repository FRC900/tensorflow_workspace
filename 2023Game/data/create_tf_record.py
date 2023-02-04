# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Convert Pascal-VOC xml annotations created by labelImg into TF Records
Example usage:
    python /home/ubuntu/tensorflow_workspace/2020Game/data/create_tf_record.py \
        --label_map_path=/home/ubuntu/tensorflow_workspace/2020Game/data/2020Game_label_map.pbtxt \
        --data_dir=/home/ubuntu/tensorflow_workspace/2020Game/data/videos \
        --alt-data_dir=/home/ubuntu/tensorflow_workspace/2019Game/data/videos \
        --output_dir=/home/ubuntu/tensorflow_workspace/2020Game/data
    python3 /home/ubuntu/tensorflow_workspace/2023Game/data/create_tf_record.py \
        --label_map_path=/home/ubuntu/tensorflow_workspace/2023Game/data/2023Game_label_map.pbtxt \
        --data_dir=/home/ubuntu/tensorflow_workspace/2023Game/data/combined_88_test \
        --alt_data_dir=/home/ubuntu/tensorflow_workspace/2023Game/data/videos \
        --alt_data_dir_2=/home/ubuntu/tensorflow_workspace/2022Game/data/test \
        --output_dir=/home/ubuntu/tensorflow_workspace/2023Game/data
"""

import hashlib
import io
import logging
import os
import random
import re
import operator
import csv
import sys
import time
import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/home/ubuntu/tensorflow_workspace/2023Game/data/combined_88_test', 'Root directory to raw dataset.')
flags.DEFINE_string('alt_data_dir', '', 'Optional second root directory to raw dataset.')
flags.DEFINE_string('alt_data_dir_2', '', 'Optional third root directory to raw dataset.')
flags.DEFINE_string('alt_data_dir_3', '', 'Optional fourth root directory to raw dataset.')
flags.DEFINE_string('output_dir', '/home/ubuntu/tensorflow_workspace/2023Game/data', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '/home/ubuntu/tensorflow_workspace/2023Game/data/2023Game_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')

FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       label_map_dict,
                       class_count_map,
                       ignore_difficult_instances=False,
                       ):
  """Convert XML derived dict to tf.Example proto.
  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.
  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
  Returns:
    example: The converted tf.Example.
  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid PNG
  """
  img_path = data['path']
  print(img_path)
  with tf.gfile.GFile(img_path, 'rb') as fid:
    print(img_path)
    try:
      encoded_image = fid.read()
    except:
      print("Error reading image: " + img_path)
      time.sleep(1)
      return None
  encoded_image_io = io.BytesIO(encoded_image)
  image = PIL.Image.open(encoded_image_io)
  if image.format != 'PNG' and image.format != 'JPEG' and image.format != 'MPO':
    print(image.format)
    raise ValueError('Image format not PNG or JPEG')
  key = hashlib.sha256(encoded_image).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  masks = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue
      difficult_obj.append(int(difficult))

      xmin = float(obj['bndbox']['xmin'])
      xmax = float(obj['bndbox']['xmax'])
      ymin = float(obj['bndbox']['ymin'])
      ymax = float(obj['bndbox']['ymax'])

      xmins.append(xmin / width)
      ymins.append(ymin / height)
      xmaxs.append(xmax / width)
      ymaxs.append(ymax / height)


      if "apriltag16h11" in obj['name']:
        obj['name'] = obj['name'].replace("apriltag16h11", "april_tag")
      class_name = obj['name']
      if (class_name in class_count_map):
          class_count_map[class_name] += 1
      else:
          class_count_map[class_name] = 1
      classes_text.append(class_name.encode('utf8'))
      try:
        class_append = label_map_dict[class_name]
      except KeyError:
        if 'april16h11' in class_name:
          # get last two characters of class name
          nums = "april_tag" + str(int(class_name[-2:]))
          class_append = label_map_dict["april_tag" + str(int(nums))] 

      classes.append(class_append)
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))
  for (xm, ym, xM, yM) in zip(xmins, ymins, xmaxs, ymaxs):    
    if xm < 0 or ym < 0 or xM > 1 or yM > 1:
      print("==========Error: bounding box out of range============")
      print(xm, ym, xM, yM)
      time.sleep(0.2)
      return None

  if image.format == 'PNG':
     image_format_str = 'png'
  elif image.format == 'JPEG':
     image_format_str = 'jpeg'
  elif image.format == 'MPO':
     image_format_str = 'jpg'

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['path'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['path'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image),
      'image/format': dataset_util.bytes_feature(image_format_str.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     examples,
                     class_map_count):
  """Creates a TFRecord file from examples.
  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    class_map_count: Stores total number of each class seen
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in enumerate(examples):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))
      xml_path = example
      print(xml_path)

      if not os.path.exists(xml_path):
        logging.warning('Could not find %s, ignoring example.', xml_path)
        continue
      with tf.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
      xml = etree.fromstring(xml_str)
      data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      try:
        tf_example = dict_to_tf_example(
            data,
            label_map_dict,
            class_map_count)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError:
        logging.warning('Invalid example: %s, ignoring.', xml_path)


def main(_):
  data_dir = FLAGS.data_dir
  alt_data_dir = FLAGS.alt_data_dir
  alt_data_dir_2 = FLAGS.alt_data_dir_2
  alt_data_dir_3 = FLAGS.alt_data_dir_3
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
  print(label_map_dict)

  logging.info('Reading from dataset.')
  examples_list = tf.gfile.Glob(os.path.join(data_dir, '*.xml'))
  examples_list += tf.gfile.Glob(os.path.join(alt_data_dir, '*.xml'))
  examples_list += tf.gfile.Glob(os.path.join(alt_data_dir_2, '*.xml'))
  examples_list += tf.gfile.Glob(os.path.join(alt_data_dir_3, '*.xml'))
  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  # TODO - should this be based on an even split of object types rather than 
  # of input images?
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.85 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, '2023Game_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, '2023Game_val.record')
  class_map_count = {}
  create_tf_record(
      train_output_path,
      FLAGS.num_shards,
      label_map_dict,
      train_examples,
      class_map_count)
  create_tf_record(
      val_output_path,
      FLAGS.num_shards,
      label_map_dict,
      val_examples,
      class_map_count)

  # print class_map_count (sorted)
  sorted_class_map_count = sorted(class_map_count.items(), key=operator.itemgetter(1))
  writer = csv.writer(sys.stdout)
  writer.writerows(sorted_class_map_count)

if __name__ == '__main__':
  tf.app.run()