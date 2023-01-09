'''
Various tests of label xml files to find errors
'''
import xml.etree.ElementTree as ET
import glob
import numpy as np
import math

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
screen_percents = []
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
        local_ar = obj_width / obj_height
        unadjusted_ar.append(local_ar)
        #if local_ar > 4:
            #print xml_file + " : " + obj.find('name').text + " : " + str(local_ar)
        #if local_ar < .2 :
            #print xml_file + " : " + obj.find('name').text + " : " + str(local_ar)

        local_adjusted_ar = (obj_width / width) / (obj_height / height)
        adjusted_ar.append (local_adjusted_ar)
        if local_adjusted_ar > 3:
            print xml_file + " : " + obj.find('name').text + " : " + str(local_adjusted_ar)
        if local_adjusted_ar < .15:
            print xml_file + " : " + obj.find('name').text + " : " + str(local_adjusted_ar)

        screen_percents.append(math.sqrt((obj_width / width) * (obj_height / height)))



#print unadjusted_ar
#print adjusted_ar

#print np.histogram(unadjusted_ar, bins='auto')
#print np.histogram(adjusted_ar, bins='auto')
print np.histogram(screen_percents, bins='auto')
