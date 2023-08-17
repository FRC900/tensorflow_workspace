import cv2
import apriltag # pip install apriltag
import os
import sys
from pascal import PascalVOC, PascalObject, BndBox, size_block # pip install pascal-voc
from pathlib import Path
from xmlformatter import Formatter

if len(sys.argv) != 2:
    print("Invalid usage, do `autolabel_apriltags.py directory` to label all images in a directory")
    sys.exit()


options = apriltag.DetectorOptions(families='tag16h5',
                                   border=1,
                                   nthreads=4,
                                   quad_decimate=0.0,
                                   quad_blur=0.5,
                                   refine_edges=True,
                                   refine_decode=True,
                                   refine_pose=False,
                                   debug=False,
                                   quad_contours=True)
detector = apriltag.Detector(options=options)
directory = sys.argv[1]

formatter = Formatter(indent="1", indent_char="\t", eof_newline=True)
pathlist = Path(directory).glob('**/*.png')
print(pathlist)
for image_path in pathlist:
    print(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    xml_path = image_path.with_suffix('.xml')
    #print(xml_path)
    # Create the xml label file if it doesn't already exist
    if (not os.path.isfile(xml_path)):
        new_ann = PascalVOC(str(image_path), path=os.path.abspath(str(image_path)), size=size_block(image.shape[1], image.shape[0], 3), objects=[])
        new_ann.save(xml_path)

    ann = PascalVOC.from_xml(xml_path)
    objs = ann.objects

    #print(objs)

    result = detector.detect(image)

    good_tags = []
    for tag in result:
        print(tag.hamming)
        #if tag.hamming == 0:
        if tag.tag_id >= 1 and tag.tag_id <= 8:
            xs = []
            ys = []
            for c in tag.corners:
                xs.append(c[0])
                ys.append(c[1])
            if max(xs) - min(xs) < 1 or max(ys) - min(ys) < 1:
                print(f'Invalid tag min/max x or y  : xs = {xs}, ys = {ys}')
                continue # invalid tag
            obj = PascalObject("april_tag_"+str(tag.tag_id), "Unspecified", truncated=False, difficult=False, bndbox=BndBox(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
            good_tags.append(obj)

    if len(good_tags) > 0:
        new_objs = objs + good_tags
        ann.objects = new_objs
        ann.save(xml_path)
        formatter.format_file(xml_path)