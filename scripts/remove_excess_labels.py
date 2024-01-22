#!/bin/bash/python3
import xml.etree.ElementTree as ET
import glob

# Gets all labels from ./2019Game_label_map.pbtxt
label_dict = []
with open("../2024Game/data/2024Game_label_map.pbtxt", "r") as f:
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

for xml_file in xml_files:
    tree = ET.parse(xml_file)
    for obj in tree.getroot().findall("object"):
        if(not obj.find("name").text in label_dict.values()):
            tree.getroot().remove(obj)
    tree.write(xml_file)
