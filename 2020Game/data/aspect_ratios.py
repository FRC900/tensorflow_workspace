'''
Various tests of label xml files to find errors
'''
import xml.etree.ElementTree as ET
import glob
import numpy as np

# Will find all xml files in the ./videos/ directory
xml_files = glob.glob("videos/*.xml")

# Given return a list of all object types with a given name
def findAllObjects(tree):
    find_str = "object"
    return tree.findall(find_str)

# Return a list of dictionaries with coords for each object
def getObjCoords(obj):
    coord_names = [ 'xmin', 'xmax', 'ymin', 'ymax' ]
    d = {}
    for coord_name in coord_names:
        d[coord_name] = int(obj.find('bndbox').find(coord_name).text)
    return d

# Search each XML file
unadjusted_ar = []
adjusted_ar = []
for xml_file in sorted(xml_files):
    tree = ET.parse(xml_file)
    width = float(tree.findall('size')[0].find('width').text)
    height = float(tree.findall('size')[0].find('height').text)
    #print str(width) + " x " + str(height)
    objects = findAllObjects(tree)
    
    for obj in objects:
        coords = getObjCoords(obj)
        obj_width = float(coords['xmax']) - float(coords['xmin'])
        obj_height = float(coords['ymax']) - float(coords['ymin'])
        unadjusted_ar.append(obj_width / obj_height)
        adjusted_ar.append ((obj_width / width) / (obj_height / height))
        if (obj_width / obj_height) > 4:
            print xml_file + " : " + obj.find('name').text + " : " + str(obj_width / obj_height)
        if (obj_width / obj_height) < .2 :
            print xml_file + " : " + obj.find('name').text + " : " + str(obj_width / obj_height)



#print unadjusted_ar
#print adjusted_ar

#print np.histogram(unadjusted_ar, bins='auto')
print np.histogram(adjusted_ar, bins='auto')
