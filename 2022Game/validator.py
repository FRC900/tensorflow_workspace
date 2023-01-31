import Augmentor
import xml.etree.ElementTree as ET
import os
import cv2
from xml.dom import minidom
import numpy as np
import time
# read in all xml files and then the image file associated with it

xmldir = os.listdir('data/test')
xmldir = [x for x in xmldir if x.endswith('.xml')]
print(xmldir)
n = 0
for file in xmldir:
    if "DSCF" in file:
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
        # check if image size matches xml file
         
        if img.shape[0] != int(root.find('size').find('height').text) or img.shape[1] != int(root.find('size').find('width').text):
            print('Image size does not match xml file')
            print(file)
            print(imgname)
            print(f"Expected: {root.find('size').find('height').text}x{root.find('size').find('width').text}")
            print(f"Actual: {img.shape[0]}x{img.shape[1]}")
            print('---------------------')
            # replace image size in xml file
            root.find('size').find('height').text = str(img.shape[0])
            root.find('size').find('width').text = str(img.shape[1])
            # write new xml file
            xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
            with open('data/test/' + file, "w") as f:
                f.write(xmlstr)
            #time.sleep(1)
        