'''
Various tests of label xml files to find errors
'''
import xml.etree.ElementTree as ET
import glob

# Will find all xml files in the ./videos/ directory
xml_files = glob.glob("/home/ubuntu/tensorflow_workspace/2024Game/data/videos/*.xml")

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
            print(20*'-')
            print(xml_file)
            print(color)
            print('loading bay left / tape coord backwards')
            print(20*'-')

    # Allow a small bit of overlap for cases where the picture is at an angle
    if (len(tape_coord) > 0) and (len(right_coord) > 0):
        if ((tape_coord[0]['xmax'] - 20) > right_coord[0]['xmin']):
            print(20*'-')
            print(xml_file)
            print(color)
            print('loading bay tape / right coord backwards')
            print(20*'-')

    if (len(left_coord) > 0) and (len(right_coord) > 0):
        if (left_coord[0]['xmax'] > right_coord[0]['xmin']):
            print(20*'-')
            print(xml_file)
            print(color)
            print('loading bay left / right coord backwards')
            print(20*'-')


# Check that power port graphics are properly sorted by height
def checkPowerPort(xml_file, tree, color):
    high_goal_coord = getObjCoords(tree, color + '_power_port_high_goal')
    low_goal_coord = getObjCoords(tree, color + '_power_port_low_goal')
    first_logo_coord = getObjCoords(tree, color + '_power_port_first_logo')
    yellow_graphics_coord = getObjCoords(tree, color + 'power_port_yellow_graphics')

    if (len(low_goal_coord) > 0) and (len(yellow_graphics_coord) > 0):
        if (low_goal_coord[0]['ymin'] < yellow_graphics_coord[0]['ymax']):
            print(20*'-')
            print(xml_file)
            print(color)
            print('power_port low goal / yellow graphics backwards')
            print(20*'-')

    if (len(yellow_graphics_coord) > 0) and (len(high_goal_coord) > 0):
        if (yellow_graphics_coord[0]['ymin'] < high_goal_coord[0]['ymax']):
            print(20*'-')
            print(xml_file)
            print(color)
            print('power_port yellow graphics / high goal backwards')
            print(20*'-')

    if (len(low_goal_coord) > 0) and (len(high_goal_coord) > 0):
        if (low_goal_coord[0]['ymin'] < high_goal_coord[0]['ymax']):
            print(20*'-')
            print(xml_file)
            print(color)
            print('power_port yellow graphics / high goal backwards')
            print(20*'-')

    if (len(low_goal_coord) > 0) :
        for flc in first_logo_coord:
            if (low_goal_coord[0]['ymin'] < flc['ymax']):
                print(20*'-')
                print(xml_file)
                print(color)
                print('power_port low goal / first logo coord backwards')
                print(20*'-')

    if (len(yellow_graphics_coord) > 0) :
        for flc in first_logo_coord:
            if (yellow_graphics_coord[0]['ymin'] < flc['ymax']):
                print(20*'-')
                print(xml_file)
                print(color)
                print('power_port yellow graphics / first logo coord backwards')
                print(20*'-')

# Search each XML fil
import os
for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
    except Exception as e:
        print(f"Failed on this {xml_file}")
        os.remove(xml_file)
        # In each, look for each object type in the
