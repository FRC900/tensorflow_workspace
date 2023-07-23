import random

import numpy as np
from object_classes import ObjectClasses
from sys import path
path.append('/home/ubuntu/tensorflow_workspace/2023Game/models')
import visualization

random.seed(0)

# detection model classes
OBJECT_CLASSES = ObjectClasses('/home/ubuntu/tensorflow_workspace/2023Game/models/yolov8/FRC2023.yaml')

# colors for per classes
COLORS = visualization.gen_colors(len(OBJECT_CLASSES))