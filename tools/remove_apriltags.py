import sys
from pascal import PascalVOC, PascalObject, BndBox # pip install pascal-voc
from pathlib import Path

print(sys.argv)
if len(sys.argv) != 3:
    print("Usage: (in current directory)\n\tremove_apriltags.py unduplicate pattern -- removes all duplicate AprilTag bounding boxes and prints out all labels with multiple of the same tag in files matching `pattern`\
\n\n\tremove_apriltags.py delete pattern -- removes ALL AprilTag detections in files matching `pattern`\n\n\n\tNOTE: pattern must end in .xml Also, * must be escaped as \\*")
    sys.exit()

operation = sys.argv[1]
glob = sys.argv[2]

def chopFilePath(filepath: str):
    return filepath[filepath.rfind("/")+1:]

pathlist = Path(".").glob(glob)
for path in pathlist:
    # because path is object not string
    xml = str(path)

    ann = PascalVOC.from_xml(xml)
    objs = ann.objects
    fakeobj = PascalObject()

    tags = {}
    final_objs = []
    
    for obj in objs:
        if "april_tag" in obj.name:
            if operation == "delete":
                continue
            if obj.name in tags.keys():
                if obj.bndbox == tags[obj.name]:
                    continue
                else:
                    print(f"DUPLICATE of {obj.name} in {path} with DIFFERENT BOUNDING BOXES!!")
            else:
                tags[obj.name] = obj.bndbox
        final_objs.append(obj)


    new_objs = final_objs
    ann.objects = new_objs
    ann.save(xml)