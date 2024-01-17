#!/bin/bash/python3
# write a script that uses pascal voc library to read all xml files in a directory and remove any annotations that are not in the pbtxt file
from pathlib import Path
from pascal import PascalVOC

# read in the pbtxt file
pbtxt_path = "/home/ubuntu/tensorflow_workspace/2024Game/data/2024Game_label_map.pbtxt"
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

#white_tape_line needs to be known as white_tape_to_wall
#red_tape_corners needs to be known as red_tape_corner
#blue_tape_corners needs to be known as blue_tape_corner


# now read all the xml files in the directory
# get a list of all the files in the target directory
target_dir = '/home/ubuntu/tensorflow_workspace/2023Game/data/combined_88_test'
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
        #another case switch scenario to check for the three values that we need to change above
        ##white_tape_line needs to be known as white_tape_to_wall
        ##red_tape_corners needs to be known as red_tape_corner
        ##blue_tape_corners needs to be known as blue_tape_corner

        if name not in classes:
            #if name == "white_tape_line":
                #replace white_tape_line with white_tape_to_wall
            #elif name == "red_tape_corners":
                #replace red_tape_corners" with red_tape_corner
            #elif name == "blue_tape_corners" with blue_tape_corner
            # remove the object
            #else:
                #to_remove.append(obj)
            if name == "white_tape_line":
                obj.name = "white_tape_to_wall"
            elif name == "red_tape_corners":
                obj.name = "red_tape_corner"
            elif name == "blue_tape_corners":
                obj.name = "blue_tape_corner"
            elif name == "ds_numbers":
                obj.name = "red_ds_numbers"
            else:
                to_remove.append(obj)
    # remove the objects
    for obj in to_remove:
        objects.remove(obj)
    # write the xml file
    voc.save(str(f))
