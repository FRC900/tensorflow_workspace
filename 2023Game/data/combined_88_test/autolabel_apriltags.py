import cv2
import apriltag # pip install apriltag
import sys
from pascal import PascalVOC, PascalObject, BndBox # pip install pascal-voc
from pathlib import Path

if len(sys.argv) != 2:
    print("Invalid usage, do `autolabel_apriltags.py directory` to label all images in a directory")
    sys.exit()

directory = sys.argv[1]

def chopFilePath(filepath: str):
    return filepath[filepath.rfind("/")+1:]

pathlist = Path(directory).glob('**/*.xml')
for path in pathlist:
    # because path is object not string
    xml = str(path)

    ann = PascalVOC.from_xml(xml)
    objs = ann.objects
    #print(objs)
    imagePath = chopFilePath(ann.filename)

    img = cv2.imread(directory+"/"+imagePath, cv2.IMREAD_GRAYSCALE)
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
    result = detector.detect(img)

    good_tags = []
    for tag in result:
        #print(tag.hamming)
        #if tag.hamming == 0:
            obj = PascalObject("april_tag_"+str(tag.tag_id), "Unspecified", truncated=False, difficult=False, bndbox=BndBox(int(tag.corners[1][0]), int(tag.corners[1][1]), int(tag.corners[3][0]), int(tag.corners[3][1])))
            good_tags.append(obj)

    new_objs = objs + good_tags
    ann.objects = new_objs
    ann.save(xml)