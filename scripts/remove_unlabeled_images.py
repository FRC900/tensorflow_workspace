#!/usr/bin/env python3

# Remove any images without a corresponding path entry in an xml file

import glob
from os import system
from pathlib import Path
from xml.etree import ElementTree as ET

xml_files = sorted(glob.glob("**/*.xml", recursive=True))

image_suffixes = set()
valid_images = set()
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    image_file = Path(tree.find('path').text)
    valid_images.add(image_file.resolve())
    image_suffixes.add(image_file.suffix)

image_files = []
for image_suffix in image_suffixes:
    print(f"Image suffix: {image_suffix}")
    image_files.extend(glob.glob(f"**/*{image_suffix}", recursive=True))
image_files = sorted(image_files)

for f in image_files:
    full_path = Path(f).resolve()
    if full_path not in valid_images:
        # print(f"Removing {full_path}")
        system(f"git rm -f '{full_path}'")