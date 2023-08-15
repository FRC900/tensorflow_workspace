#!/usr/bin/env python3
# Checks that xmin<xmax, ymin<ymax for all labels in 
from pathlib import Path

from pascal import PascalVOC, BndBox
from sys import argv

# now read all the xml files in the directory
# get a list of all the files in the target directory
files = Path(argv[1]).glob('*.xml')
files = [f for f in files if f.is_file()]
print(files)

# loop through the files
for f in files:
    # read the xml file
    voc = PascalVOC.from_xml(str(f))

    update_file = False
    # loop through the objects
    for obj in voc.objects:
        x_min = min(obj.bndbox.xmin, obj.bndbox.xmax)
        x_max = max(obj.bndbox.xmin, obj.bndbox.xmax)
        y_min = min(obj.bndbox.ymin, obj.bndbox.ymax)
        y_max = max(obj.bndbox.ymin, obj.bndbox.ymax)
        if (x_min != obj.bndbox.xmin) or (x_max != obj.bndbox.xmax) or (y_min != obj.bndbox.ymin) or (y_max != obj.bndbox.ymax):
            print(f'file {str(f)}, bnd box difference for {obj.name}')
            obj.bndbox = BndBox(xmin=x_min,ymin=y_min, xmax=x_max, ymax=y_max)
            update_file = True

    # write the xml file if it has changed
    if update_file:
        voc.save(str(f))
