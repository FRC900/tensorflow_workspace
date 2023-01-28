# write a script that goes through all of the .xml files in a target directory and replaces the path element with the correct path

import os
import xml.etree.ElementTree as ET

# set the target directory
target_dir = 'videos'

# get a list of all the files in the target directory
files = os.listdir(target_dir)
files = [f for f in files if f.endswith('.xml')]
name = "chris"
# loop through the files
for f in files:
    # get the full path to the file
    full_path = os.path.join(target_dir, f)
    full_path = os.path.abspath(full_path)
    # replace the text between the /home/ and /tensorflow_workspace with ubuntu

    #print(full_path)
    
    # parse the xml
    tree = ET.parse(full_path)
    root = tree.getroot()
    # loop through the elements
    
    for elem in root.iter():
        # if the element is a path element
        if elem.tag == 'path' or elem.tag == 'filename':
            full_path_xml = elem.text
            if elem.tag == 'filename':
                full_path_xml = os.path.join(target_dir, elem.text)
            else:
                full_path_xml = os.path.join(target_dir, f)
            full_path_xml = os.path.abspath(full_path_xml)
            full_path_xml = full_path_xml.replace(name, "ubuntu")
            print(full_path_xml)
            # set the text to the full path
            elem.text = full_path_xml
    # write the xml
    tree.write(full_path)
    