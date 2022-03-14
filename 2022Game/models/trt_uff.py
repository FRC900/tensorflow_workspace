"""
Adaped from	https://github.com/AastaNV/TRT_object_detection.git  
A faster way to optimize models to run on the Jetson
This script has 2 parts. First is to convert the model to UFF format and then
optimize that using tensorRT.  This produces a .bin file.
The .bin file is then loaded and used to run inference on a video.
"""
import os
import sys
import cv2
import time
import ctypes
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda


import tensorrt as trt
#from config import model_ssd_inception_v2_coco_2017_11_17 as model
#from config import model_ssd_mobilenet_v1_coco_2018_01_28 as model
#from config import model_ssd_mobilenet_v2_coco_2018_03_29 as model
#from config import retinanet_mobilenet_v2_400x400 as model
from config import model_ssd_mobilenet_v2_512x512 as model
#from config import model_ssd_mobilenet_v3 as model
from visualization import BBoxVisualization
import timing
from object_detection.utils import label_map_util

ctypes.CDLL("/home/ubuntu/TensorRT/build/libnvinfer_plugin.so")
#CLASS_LABELS = class_labels.CLASSES_LIST


# initialize
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)


# compile model into TensorRT
# This is only done if the output bin file doesn't already exist
# TODO - replace this with the MD5 sum check we have for the other TRT detection
if not os.path.isfile(model.TRTbin):
    import uff
    import graphsurgeon as gs
    dynamic_graph = model.add_plugin(gs.DynamicGraph(model.path))
    uff_model = uff.from_tensorflow(dynamic_graph.as_graph_def(), model.output_name, output_filename='tmp.uff')

    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 29
        builder.max_batch_size = 1
        #builder.fp16_mode = True

        parser.register_input('Input', model.dims)
        parser.register_output('MarkOutput_0')
        parser.parse('tmp.uff', network)
        engine = builder.build_cuda_engine(network)

        buf = engine.serialize()
        with open(model.TRTbin, 'wb') as f:
            f.write(buf)


# Start of inference code
# create engine
with open(model.TRTbin, 'rb') as f:
    buf = f.read()
    engine = runtime.deserialize_cuda_engine(buf)


# create buffers
host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    host_mem = cuda.pagelocked_empty(size, np.float32)
    cuda_mem = cuda.mem_alloc(host_mem.nbytes)

    bindings.append(int(cuda_mem))
    if engine.binding_is_input(binding):
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
    else:
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)
context = engine.create_execution_context()

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/ubuntu/tensorflow_workspace/2022Game/data', '2022Game_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

category_dict = {0: 'background'}
for k in category_index.keys():
    category_dict[k] = category_index[k]['name']
viz = BBoxVisualization(category_dict)

t = timing.Timings()

PATH_TO_TEST_IMAGES_DIR = '/home/ubuntu/tensorflow_workspace/2022Game/data/videos'
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2022 Field Tour Video Alliance Area.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2022 Field Tour Video Hangar.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2022 Field Tour Video Hub.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, '2022 Field Tour Video Terminal.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'FRCTwitchVision.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Filming Videos_ds0_2021-11-19 09-32-48.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Filming Videos_ds0_2021-11-19 09-33-03.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Qualification 14 - 2022 Week 0.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'TraversalClimb125.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Y2Mate.is - 2022 FIRST Robotics Competition RAPID REACT Game Animation-LgniEjI9cCM-1080p-1641829421487.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'Y2Mate.is - The Robonauts 118 Everybot 2022-HdQ-mWPG9GY-720p-1642873990701.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_230907441.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_230927319.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_231105193.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_231126567.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_231135548.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_231355465.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_230158009.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_230311691.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_230407177.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_230551890.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'PXL_20220215_230749699.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'week0_1.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'week0_2.mp4'))
#cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'IMG_0023-1920.mp4'))
cap = cv2.VideoCapture(os.path.join(PATH_TO_TEST_IMAGES_DIR, 'IMG_0034-1920.mp4'))


# inference
#TODO enable video pipeline
#TODO using pyCUDA for preprocess
#ori = cv2.imread(sys.argv[1])
ret, ori = cap.read()
print( ori.shape)
#vid_writer = cv2.VideoWriter(os.path.join(PATH_TO_TEST_IMAGES_DIR, '5172_POV-Great_Northern_2020_Quals_60_annotated.mp4'), cv2.VideoWriter_fourcc(*"FMP4"), 30., (ori.shape[1], ori.shape[0]))

while(True):
  t.start('frame')
  t.start('vid')
  ret, ori = cap.read()
  t.end('vid')
  if not ret:
    break

  next_frame = False
  while (not next_frame):
    t.start('cv')
    image = cv2.resize(ori, (model.dims[2],model.dims[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (2.0/255.0) * image - 1.0 # Convert from 0to255 to -1to1 range for input values
    image = image.transpose((2, 0, 1))
    np.copyto(host_inputs[0], image.ravel())
    t.end('cv')
    
    t.start('inference')
    start_time = time.time()
    cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_outputs[1], cuda_outputs[1], stream)
    cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
    stream.synchronize()
    t.end('inference')
    
    t.start('viz')
    output = host_outputs[0]
    height, width, channels = ori.shape
    boxes = []
    confs = []
    clss = []
    for i in range(int(len(output)/model.layout)):
        prefix = i*model.layout
        '''
        index = int(output[prefix+0])
        label = int(output[prefix+1])
        conf  = output[prefix+2]
        xmin  = int(output[prefix+3]*width)
        ymin  = int(output[prefix+4]*height)
        xmax  = int(output[prefix+5]*width)
        ymax  = int(output[prefix+6]*height)
        if conf > 0.2:
            print("Detected {} with confidence {}".format(CLASS_LABELS[label], "{0:.0%}".format(conf)))
            cv2.rectangle(ori, (xmin,ymin), (xmax, ymax), (0,0,255),3)
            cv2.putText(ori, CLASS_LABELS[label],(xmin+10,ymin+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        '''
    
        boxes.append([output[prefix+4], output[prefix+3], output[prefix+6], output[prefix+5]])
        clss.append(int(output[prefix+1]))
        confs.append(output[prefix+2])
    
    viz.draw_bboxes(ori, boxes, confs, clss, 0.42)
    
    #cv2.imwrite("result.jpg", ori)
    cv2.imshow("result", ori)
    #vid_writer.write(ori)
    t.end('viz')
    key = cv2.waitKey(1) & 0x000000FF
    next_frame = True
  t.end('frame')
