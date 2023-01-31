import xml.etree.ElementTree as ET
import os
import cv2
from xml.dom import minidom
import numpy as np
import time
import threading

def check_bounding_box(file):
    print(file)
    tree = ET.parse('data/test/' + file)
    root = tree.getroot()
    # find image name in xml file
    imgname = root.find('filename').text
    img = cv2.imread('data/test/' + imgname)
    # find all bounding boxes in xml file
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        # check if bounding box is within image
        if xmin < 0 or ymin < 0 or xmax > img.shape[1] or ymax > img.shape[0]:
            print('Bounding box is outside image')
            print(file)
            print(imgname)
            print(xmin, ymin, xmax, ymax)
            print(img.shape)
            print('---------------------')
            time.sleep(1)

# read in all xml files and then the image file associated with it
xmldir = os.listdir('data/test')
xmldir = [x for x in xmldir if x.endswith('.xml')]
print(xmldir)

threads = []
max_threads = 20

for file in xmldir:
    while threading.active_count() > max_threads:
        time.sleep(0.1)
    t = threading.Thread(target=check_bounding_box, args=(file,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()
