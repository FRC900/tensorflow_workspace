#!/bin/bash/python3

# Search through labels, find and remove any labeled objects that 
# are not in the current year's set of objects
import xml.etree.ElementTree as ET
import glob
import yaml

with open("/home/ubuntu/900RobotCode/zebROS_ws/src/tf_object_detection/src/FRC2025.yaml") as file:
    label_dict = yaml.safe_load(file)['names']

print(label_dict)

# Will find all xml files in the current directory and all subdirectories
xml_files = sorted(glob.glob("**/*.xml", recursive=True))

for xml_file in xml_files:
    changed = False
    print(f'xml_file : {xml_file}')
    tree = ET.parse(xml_file)
    for obj in tree.getroot().findall("object"):
        if obj.find("name").text not in label_dict.values():
            print(f'\t...Removing {obj.find("name").text}')
            tree.getroot().remove(obj)
            changed = True
    if changed:
        tree.write(xml_file)
