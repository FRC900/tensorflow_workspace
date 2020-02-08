'''
Various tests of label xml files to find errors
'''
import xml.etree.ElementTree as ET
import glob

# Will find all xml files in the ./videos/ directory
xml_files = glob.glob("videos/*.xml")

# Max number of a each object seen in an image
max_obj_counts = {
'yellow_control_panel_light' : 2,
'blue_power_port_high_goal' : 1,
'red_robot' : 3,
'red_shield_generator_light' : 1,
'blue_shield_generator_light' : 1,
'blue_robot' : 3,
'red_loading_bay_left_graphics' : 1,
'blue_black_shield_generator_floor_intersection' : 1,
'shield_generator_floor_center_intersection' : 1,
'red_loading_bay_tape' : 1,
'blue_power_port_low_goal' : 1,
'red_black_shield_generator_floor_intersection' : 1,
'red_power_port_high_goal' : 1,
'red_loading_bay_right_graphics' : 1,
'blue_loading_bay_tape' : 1,
'control_panel_light' : 2,
'blue_power_port_first_logo' : 2,
'blue_loading_bay_left_graphics' : 1,
'blue_loading_bay_right_graphics' : 1,
'red_shield_pillar_intersection' : 2,
'blue_shield_pillar_intersection' : 2,
'red_ds_light' : 3,
'ds_light' : 5,
'red_blue_black_shield_generator_floor_intersection' : 2,
'red_power_port_low_goal' : 1,
'red_power_port_first_logo' : 2,
'color_wheel' : 2,
'shield_generator_backstop' : 4,
'power_port_yellow_graphics' : 1,
'blue_ds_light' : 3,
'ds_numbers' : 4,
'power_port_yellow_graphics': 4,
'power_port_first_logo': 4,
}

# Given return a list of all object types with a given name
def findAllObjectsNamed(tree, obj_name):
    find_str = "object[name='" + obj_name +"']"
    return tree.findall(find_str)

# Return a list of dictionaries with coords for each object
def getObjCoords(tree, obj_name):
    coord_names = [ 'xmin', 'xmax', 'ymin', 'ymax' ]
    ret = []
    objs = findAllObjectsNamed(tree, obj_name)
    for obj in objs:
        d = {}
        for coord_name in coord_names:
            d[coord_name] = int(obj.find('bndbox').find(coord_name).text)
        ret.append(d)
    return ret

# Check that the x coords of loading bay graphics are in the proper order
def checkLoadingBay(xml_file, tree, color):
    left_coord = getObjCoords(tree, color + '_loading_bay_left_graphics')
    tape_coord = getObjCoords(tree, color + '_loading_bay_tape')
    right_coord = getObjCoords(tree, color + '_loading_bay_right_graphics')

    if (len(left_coord) > 0) and (len(tape_coord) > 0):
        if (left_coord[0]['xmax'] > tape_coord[0]['xmin']):
            print 20*'-'
            print xml_file
            print color
            print 'loading bay left / tape coord backwards'
            print 20*'-'

    # Allow a small bit of overlap for cases where the picture is at an angle
    if (len(tape_coord) > 0) and (len(right_coord) > 0):
        if ((tape_coord[0]['xmax'] - 20) > right_coord[0]['xmin']):
            print 20*'-'
            print xml_file
            print color
            print 'loading bay tape / right coord backwards'
            print 20*'-'

    if (len(left_coord) > 0) and (len(right_coord) > 0):
        if (left_coord[0]['xmax'] > right_coord[0]['xmin']):
            print 20*'-'
            print xml_file
            print color
            print 'loading bay left / right coord backwards'
            print 20*'-'

# Search each XML file
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    # In each, look for each object type in the
    # XML. Make sure there aren't more labeled
    # images than could possibly be seen in any one image
    for obj in max_obj_counts:
        r = findAllObjectsNamed(tree, obj)
        if (len(r) > max_obj_counts[obj]):
            print 20*'-'
            print obj
            print xml_file
            print len(r)
            print 20*'-'

    checkLoadingBay(xml_file, tree, 'red')
    checkLoadingBay(xml_file, tree, 'blue')


