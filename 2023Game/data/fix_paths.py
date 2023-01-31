# write a script that goes through all of the .xml files in a target directory and replaces the path element with the correct path

import os
import xml.etree.ElementTree as ET

# set the target directory
target_dir = 'combined_88_test'

# get a list of all the files in the target directory
files = os.listdir(target_dir)
files = [f for f in files if f.endswith('.xml')]
names = ["nathan", "aws.admin"]
# read each xml file as string and replace /frc/ with ""
'''
for f in files:
    full_path = os.path.join(target_dir, f)
    full_path = os.path.abspath(full_path)
    with open(full_path, 'r') as file :
        filedata = file.read()
    filedata = filedata.replace("//", "/")
    with open(full_path, 'w') as file:
        file.write(filedata)
'''
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
    # get path and filename elements to variable
    filename_elem = root.find('filename')
    path_elem = root.find('path')

    if "/" in filename_elem.text:
        path_elem.text = filename_elem.text
        # get the filename from the path
        filename_elem.text = os.path.basename(filename_elem.text)
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