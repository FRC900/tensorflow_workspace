# write a script that goes through all of the .xml files in a target directory and replaces the path element with the correct path

import os
import xml.etree.ElementTree as ET
import cv2
# set the target directory
target_dir = '../../2022Game/data/test'

# get a list of all the files in the target directory
files = os.listdir(target_dir)
files = [f for f in files if f.endswith('.xml')]
names = ["nathan", "aws.admin"]

# read each xml file as string and replace /frc/ with ""
for f in files:
    full_path = os.path.join(target_dir, f)
    full_path = os.path.abspath(full_path)
    with open(full_path, 'r') as file :
        filedata = file.read()
    for name in names:
        filedata = filedata.replace(name, "ubuntu")
    filedata = filedata.replace("april16h11_0", "april_tag_")
    with open(full_path, 'w') as file:
        file.write(filedata)
exit()

prefix = "/home/ubuntu/tensorflow_workspace/2023Game/data/videos/"
for f in files:
    full_path = os.path.join(target_dir, f)
    full_path = os.path.abspath(full_path)
    tree = ET.parse(full_path)
    root = tree.getroot()
    filename_elem = root.find('filename')
    path_elem = root.find('path')
    to_be_path = prefix + filename_elem.text
    assert to_be_path
    print(f"Path: {to_be_path}")
    path_elem.text = to_be_path
    tree.write(full_path)


exit()
# loop through the files
for f in files:
    # get the full path to the file
    full_path = os.path.join(target_dir, f)
    full_path = os.path.abspath(full_path)
    full_path_ubuntu = full_path.replace("chris", "ubuntu")
    print(full_path_ubuntu)
    # replace the text between the /home/ and /tensorflow_workspace with ubuntu

    #print(full_path)
    # filename = short 
    # path = full
    # both for image

    # parse the xml
    tree = ET.parse(full_path)
    root = tree.getroot()
    # loop through the elements
    # get path and filename elements to variable
    filename_elem = root.find('filename')
    path_elem = root.find('path')
    #print(f"filename_elem: {path_elem.text}")
    #print(f"path_elem: {filename_elem.text}")
    assert "/" in filename_elem.text and "/" in path_elem.text
    assert path_elem.text.endswith(".xml")
    
    if filename_elem.text.endswith(".jpg") or filename_elem.text.endswith(".png"):
        jpg_abs_path = filename_elem.text
    elif path_elem.text.endswith(".jpg") or filename_elem.text.endswith(".png"):
        jpg_abs_path = path_elem.text
    else:
        assert False, "No jpg found"

    # read in the image and verify it exists
    # img = cv2.imread(jpg_abs_path.replace("ubuntu", "chris"))
    # assert img is not None, f"Image not found: {jpg_abs_path}"
    # both above asserts have been proven true

    print(f"Path: {jpg_abs_path}")
    path_elem.text = filename_elem.text

    # get just the filename from the path with os.path.basename
    filename = os.path.basename(jpg_abs_path)
    print(f"Filename: {filename}")
    
    filename_elem.text = filename
    print()
    # write the xml
    tree.write(full_path)
    

'''
    for elem in root.iter():
        # if the element is a path element
        if elem.tag == 'path' or elem.tag == 'filename':
            full_path_xml = elem.text
            if elem.tag == 'filename':
                full_path_xml = os.path.join(target_dir, elem.text)
            else:
                full_path_xml = os.path.join(target_dir, f)
            full_path_xml = os.path.abspath(full_path_xml)
            for name in names:
                full_path_xml = full_path_xml.replace(name, "ubuntu")
            print(full_path_xml)
            # set the text to the full path
            elem.text = full_path_xml
            '''