#!/bin/bash/python3
# write a script that uses pascal voc library to read all xml files in a directory and remove any annotations that are not in the pbtxt file
from pathlib import Path

import cv2
from pascal import PascalVOC

# read in the pbtxt file
pbtxt_path = "/home/nathan/tensorflow_workspace/2024Game/data/2024Game_label_map.pbtxt"
def read_label_map(label_map_path):

    item_id = None
    item_name = None
    items = {}
    
    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_name] = item_id
                item_id = None
                item_name = None

    return items

classes = [i for i in read_label_map(pbtxt_path)]
print(classes)

# now read all the xml files in the directory
# get a list of all the files in the target directory
target_dir = '/home/nathan/tensorflow_workspace/2023Game/data/videos'
files = Path(target_dir).glob('*.xml')
files = [f for f in files if f.is_file()]
print(files)

# loop through the files
for f in files:
    # read the xml file
    voc = PascalVOC.from_xml(str(f))
    # get the objects
    objects = voc.objects

    # loop through the objects
    to_remove = []
    for obj in objects:
        # get the name
        name = obj.name
        # if the name is not in the classes list
        if name not in classes:
            # remove the object
            to_remove.append(obj)
    # remove the objects
    for obj in to_remove:
        objects.remove(obj)
    # write the xml file
    voc.save(str(f))
