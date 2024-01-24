# write a script that goes through all of the .xml files in a target directory and replaces the path element with the correct path

import os
import xml.etree.ElementTree as ET
import cv2

# set the target directory
target_dir = '/home/ubuntu/tensorflow_workspace/2024Game/data/videos'

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
    try:
        tree = ET.parse(full_path)
    except:
        print(f"failed to parse xml file: {full_path}")
        #os.remove(full_path)
        continue
    root = tree.getroot()
    # loop through the elements
    # get path and filename elements to variable
    filename_elem = root.find('filename')
    path_elem = root.find('path')

    if "/" in filename_elem.text:
        path_elem.text = filename_elem.text
        # get the filename from the path
        filename_elem.text = os.path.basename(filename_elem.text)
    # attempt to read the image with opencv
    try:
        img = cv2.imread(path_elem.text)
    except:
        print(f"failed to read image path: {path_elem.text}")
        #os.remove(full_path)
        continue

    # validate PascalVOC xml
    assert root.tag == 'annotation' or root.attrib['verified'] == 'yes', "PASCAL VOC does not contain a root element" # Check if the root element is "annotation"
    assert len(root.findtext('folder')) > 0, "XML file does not contain a 'folder' element"
    assert len(root.findtext('filename')) > 0, "XML file does not contain a 'filename'"
    assert len(root.findtext('path')) > 0, "XML file does not contain 'path' element"
    assert len(root.find('source')) == 1 and len(root.find('source').findtext('database')) > 0, "XML file does not contain 'source' element with a 'database'"
    assert len(root.find('size')) == 3, "XML file doesn not contain 'size' element"
    assert root.find('size').find('width').text and root.find('size').find('height').text and root.find('size').find('depth').text, "XML file does not contain either 'width', 'height', or 'depth' element"
    assert root.find('segmented').text == '0' or len(root.find('segmented')) > 0, "'segmented' element is neither 0 or a list"
    # assert len(root.findall('object')) > 0, "XML file contains no 'object' element" # Check if the root contains zero or more 'objects'

    required_objects = ['name', 'pose', 'truncated', 'difficult', 'bndbox'] # All possible meta-data about an object
    for obj in root.findall('object'):
        assert len(obj.findtext(required_objects[0])) > 0, "Object does not contain a parameter 'name'"
        assert len(obj.findtext(required_objects[1])) > 0, "Object does not contain a parameter 'pose'"
        assert int(obj.findtext(required_objects[2])) in [0, 1], "Object does not contain a parameter 'truncated'"
        assert int(obj.findtext(required_objects[3])) in [0, 1], "Object does not contain a parameter 'difficult'"
        assert len(obj.findall(required_objects[4])) > 0, "Object does not contain a parameter 'bndbox'"
        for bbox in obj.findall(required_objects[4]):
            xmin = bbox.findtext('xmin')
            ymin = bbox.findtext('ymin')
            xmax = bbox.findtext('xmax')
            ymax = bbox.findtext('ymax')
            assert xmin and ymin and xmax and ymax, "Object does not contain either 'xmin', 'ymin', 'xmax', or 'ymax' element"
            # check if the bounding box is valid using the image size
            assert int(xmin) >= 0, "xmin is less than 0"
            assert int(ymin) >= 0, "ymin is less than 0"
            assert int(xmax) <= int(root.find('size').findtext('width')), "xmax is greater than image width"
            assert int(ymax) <= int(root.find('size').findtext('height')), "ymax is greater than image height"
            assert int(xmin) < int(xmax), "xmin is greater than xmax"
            assert int(ymin) < int(ymax), "ymin is greater than ymax"
                        

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
