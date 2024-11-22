#!/usr/bin/env python3
#Removes unlabeled images

from os import listdir, path, remove
from sys import argv 
from pathlib import Path

files = listdir(argv[1])

xml_files = [f for f in files if Path(f).suffix == '.xml']
xml_file_stems = [Path(f).with_suffix('') for f in xml_files]

files_to_remove = [f for f in files if Path(f).with_suffix('') not in xml_file_stems] 

unlabeled_img = []
for f in files_to_remove:
    path_to_remove = Path(argv[1]) / f
    if not path.isfile(path_to_remove.with_suffix('.xml')):
        unlabeled_img.append(path_to_remove)


#!/bin/bash/python3
#Removes excess labels
import xml.etree.ElementTree as ET
import glob

#Switch file names to get labels from different years
label_dict = []
with open("2024Game/data/2024Game_label_map.pbtxt", "r") as f:
    data = f.read().splitlines()
    print(data)
    line = 0
    while line < len(data):
        data[line+2].split(" ")[-1]
        label_dict.append((int(data[line+1].split(" ")[-1]), data[line+2].split(" ")[-1][1:-1]))
        line += 5

label_dict = dict(label_dict) # An enumerated dict

# Will find all xml files in the ./videos/ directory
xml_files = glob.glob("videos/*.xml")

unlabeled_xml = []
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    for obj in tree.getroot().findall("object"):
        if(not obj.find("name").text in label_dict.values()):
            unlabeled_xml.append(obj)
    tree.write(xml_file)

for xml in unlabeled_xml:
    print(xml)

for img in unlabeled_img:
    print(img)