import matplotlib
matplotlib.use('TkAgg')
import tensorflow as tf 
from object_detection.anchor_generators.multiple_grid_anchor_generator import create_ssd_anchors
from object_detection.models.ssd_mobilenet_v2_feature_extractor_test import SsdMobilenetV2FeatureExtractorTest
from object_detection.models import ssd_mobilenet_v2_feature_extractor

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np


def get_feature_map_shapes(image_height, image_width):
    """
    :param image_height: height in pixels
    :param image_width: width in pixels
    :returns: list of tuples containing feature map resolutions
    """
    
    feature_extractor = SsdMobilenetV2FeatureExtractorTest('test_has_fused_batchnorm0')._create_feature_extractor(depth_multiplier=1, pad_to_multiple=1)
    image_batch_tensor = tf.zeros([1, image_height, image_width, 1])
    
    return [tuple(feature_map.get_shape().as_list()[1:3])
            for feature_map in feature_extractor.extract_features(image_batch_tensor)]

    

def get_feature_map_anchor_boxes(feature_map_shape_list, **anchor_kwargs):
    """
    :param feature_map_shape_list: list of tuples containing feature map resolutions
    :returns: dict with feature map shape tuple as key and list of [ymin, xmin, ymax, xmax] box co-ordinates
    """
    anchor_generator = create_ssd_anchors(**anchor_kwargs)

    anchor_box_lists = anchor_generator.generate(feature_map_shape_list)
    
    feature_map_boxes = {}

    with tf.Session() as sess:
        for shape, box_list in zip(feature_map_shape_list, anchor_box_lists):
            feature_map_boxes[shape] = sess.run(box_list.data['boxes'])
            
    return feature_map_boxes


def draw_boxes(boxes, figsize, nrows, ncols, grid=(0,0)):

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize) 

    for ax, box in zip(axes.flat, boxes):
        ymin, xmin, ymax, xmax = box
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, 
                                fill=False, edgecolor='red', lw=2))

        # add gridlines to represent feature map cells
        ax.set_xticks(np.linspace(0, 1, grid[0] + 1), minor=True)
        ax.set_yticks(np.linspace(0, 1, grid[1] + 1), minor=True)
        ax.grid(True, which='minor', axis='both')
              
    fig.tight_layout()
    
    return fig

feature_map_boxes = get_feature_map_anchor_boxes(feature_map_shape_list=get_feature_map_shapes(300, 300))
fig = draw_boxes(feature_map_boxes[3,3], figsize=(12,16), nrows=9, ncols=6, grid=(3,3))
fig.show()

