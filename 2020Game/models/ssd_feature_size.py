"""
Code which calculates and displays the feature map sizes for an SSD detector
using a given input size. Pass in the input image width and height as command
line options - 

python ./ssd_feature_size.py 300 300
"""
import sys
import tensorflow as tf
from object_detection.anchor_generators.multiple_grid_anchor_generator import create_ssd_anchors
from object_detection.models.ssd_mobilenet_v2_feature_extractor_test import SsdMobilenetV2FeatureExtractorTest

feature_extractor = SsdMobilenetV2FeatureExtractorTest('test_has_fused_batchnorm0')._create_feature_extractor(depth_multiplier=1, pad_to_multiple=1)
image_batch_tensor = tf.zeros([1, int(sys.argv[1]), int(sys.argv[2]), 1])
print([tuple(feature_map.get_shape().as_list()[1:3])
       for feature_map in feature_extractor.extract_features(image_batch_tensor)])
