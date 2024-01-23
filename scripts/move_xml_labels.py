#!/usr/bin/env python3

'''
xml labels for giant FRC2024 dataset are in one directiory while the images
are split into different train / test / val dirs. Move them to the 
same dir as their images
'''

from os import listdir, system
from sys import argv 
from pathlib import Path

from pascal import PascalVOC

files = sorted(listdir(argv[1]))

xml_files = [Path(argv[1]) / Path(f) for f in files if Path(f).suffix == '.xml']

for xml_file in xml_files:
    # open xml as pascal-voc object
    pascal_voc = PascalVOC.from_xml(xml_file)
    # get the directory of the image
    image_path = Path(pascal_voc.path).parent

    # Move the xml file to its image's directory
    # print(f'xml_file: {xml_file}, image_path: {image_path}')
    print(f'git mv {xml_file} {image_path / xml_file.name}')
    # system(f'git mv {xml_file} {image_path / xml_file.name}')