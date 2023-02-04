# read in all files in a directory that have a string
# make a pascale voc xml file for each image with no objects

import os
import cv2
import xml.etree.ElementTree as ET

read_dir = "/home/chris/tensorflow_workspace/2023Game/data/combined_88_test"
to_have = ["WARRIORS", "Minnesota"]

files = os.listdir(read_dir)
files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]
# make sure it has to have at least one of the strings
files = [f for f in files if any([t in f for t in to_have])]

for f in files:

    # get the full path to the file
    full_path = os.path.join(read_dir, f)
    full_path = os.path.abspath(full_path)
    full_path_ubuntu = full_path.replace("chris", "ubuntu")
    print(full_path)

    # read in the image and verify it exists
    img = cv2.imread(full_path)
    assert img is not None, f"Image not found: {full_path}"

    # get the width and height of the image
    height, width, _ = img.shape
    # make xml file
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "images"
    ET.SubElement(root, "filename").text = f
    ET.SubElement(root, "path").text = full_path_ubuntu
    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"
    # write the xml file
    tree = ET.ElementTree(root)
    xml_path = os.path.join(read_dir, f.replace(".jpg", ".xml").replace(".png", ".xml"))
    tree.write(xml_path)
    