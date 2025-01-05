#!/bin/bash/python3

import glob
from os import system
import xml.etree.ElementTree as ET

# Look through list of xml files
# Find any xml files that have zero objects in them
# Remove the xml file and the corresponding image file

xml_files = sorted(glob.glob("**/*.xml", recursive=True))

for xml_file in xml_files:
    tree = ET.parse(xml_file)
    num_objects = len(tree.getroot().findall("object"))
    # print(f'xml_file : {xml_file} num_objects: {num_objects}')
    if num_objects == 0:
        image_file = tree.find('path').text
        # print(f'Removing {xml_file} and {image_file}')
        system(f'git rm -f "{xml_file}" "{image_file}"')