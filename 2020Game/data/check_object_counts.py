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

# Search each XML file
# In each, look for each object type in the
# XML. Make sure there aren't more labeled
# images than could possibly be seen in any one image
for xml_file in xml_files:
    tree = ET.parse(xml_file)
    for obj in max_obj_counts:
        findStr = "object[name='" + obj +"']"
        r = tree.findall(findStr)
        if (len(r) > max_obj_counts[obj]):
            print 20*'-'
            print obj
            print xml_file
            print len(r)
            print 20*'-'



