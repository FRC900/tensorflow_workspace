import Augmentor
import xml.etree.ElementTree as ET
import os
# import openCV
import cv2
# import minidom
from xml.dom import minidom
import shutil
from collections import namedtuple
import random
import time

# make namped tuple for bounding box
BBox = namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])
Tag_BBox = namedtuple('Tag_BBox', ['BBox', 'id']) # BBox should be instance of BBox


class ApriltagTrainer:
    
    '''
    tag_path: path to the apriltag folder containing the images of tags in png or jpg format
    data_dir: path to the directory where images of field and xml annotations are stored
              will modify the xml files in this directory and images 
    returns: nothing
    '''
    def __init__(self, tag_path, data_dir, new_dir=False) -> None:
        self.new_dir = new_dir
        self.new_dir_name = "data/test"
        self.tags_to_add = [1, 1, 1, 1, 2, 2, 2, 3, 3] # ratio of how many tags to add to each image
        self.data_dir = data_dir
        self.tag_path = tag_path
        print(f"Tag path: {tag_path} \nData path: {data_dir}")
        self.p = Augmentor.Pipeline(self.tag_path)    
        self.p.rotate_random_90(probability=0.75)
        self.p.zoom(probability=0.5, min_factor=0.7, max_factor=1.4)
        self.p.skew(probability=0.75, magnitude=0.5)
        self.p.random_distortion(probability=0.5, grid_width=5, grid_height=5, magnitude=3)
        self.p.random_brightness(probability=0.7, min_factor=0.3, max_factor=1.5)
        
    '''
    data_dir: path to the directory containing the images and xml files of previously trained images
    n: optional parameter to specify the number of images to be generated
        overrides using the number of xml files in the data_dir
    returns: nothing
    '''
    def augment(self, n=None):
        # don't have great reason for this, just like to have the directory clean
        # check if tag_path/output exists, if so delete it
        if os.path.exists(os.path.join(self.tag_path, "output")):
            print("Deleting old output directory")
            # delete the output directory and all its contents
            shutil.rmtree(os.path.join(self.tag_path, "output"))
            # make a new output directory, errors without this
            os.mkdir(os.path.join(self.tag_path, "output"))

        if n:
            num_samples = n
        else:
            dirlist = os.listdir(self.data_dir)
            # remove all non .xml files
            dirlist = [x for x in dirlist if x[-4:] == ".xml"]
            num_samples = int(len(dirlist) * 2.2) # will add 2 images per xml file on average 
        print(f"\nNumber of samples generated: {num_samples}")
        self.p.sample(num_samples)
    
    '''
    Usage: Uses the augmented images from tag_path/output and goes through each xml file in the 
    data_dir and the corresponding image. It then adds the apriltag to the image and modifies the
    xml file to reflect the new bounding box coordinates. 
    It then saves the new image and xml file in the data_dir, _overwriting_.
    '''
    def generate_xml(self):
        # get a list of all the apriltag images
        aprildirlist = os.listdir(os.path.join(self.tag_path, "output"))
        # shuffle list
        random.shuffle(aprildirlist)

        datadirlist = os.listdir(self.data_dir)
        datadirlist = [x for x in datadirlist if x[-4:] == ".xml"]
        for xml_file in datadirlist:
            print("\n\n\n")
            # print(f"Processing {xml_file}")
            # get the image name from the xml file data
            tree = ET.parse(os.path.join(self.data_dir, xml_file))
            root = tree.getroot()
            image_name = root.find('filename').text
            # print(f"Processing {image_name}")
            image_path = os.path.join(self.data_dir, image_name)
            # read the image
            image = cv2.imread(image_path)
            # get the image dimensions
            height, width, channels = image.shape
            bounding_boxes = []
            # find all bounding boxes in the xml file
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)
                bounding_boxes.append(BBox(xmin, ymin, xmax, ymax))

            tags_to_write = []
            to_add = random.choice(self.tags_to_add)
            print(f"Number of tags to add: {to_add}")
            
            for _ in range(to_add):
                apriltag = aprildirlist.pop()
                # read an augemented apriltag image
                tag_image = cv2.imread(os.path.join(self.tag_path, "output", apriltag))
                # get tag id, 5 characters before the . in the first instance of .png
                tag_id = int(apriltag[apriltag.find(".png")-5:apriltag.find(".png")])
                # get the tag dimensions
                tag_height, tag_width, _ = tag_image.shape
                print(f"Tag id: {tag_id}, file name {apriltag} \n Tag height: {tag_height}\n Tag width: {tag_width}")
                aligned = False
                # print("Trying to find a good spot for the tag")
                while not aligned:
                    # get a random x and y coordinate for the tag
                    x = random.randint(0, width - tag_width)
                    y = random.randint(0, height - tag_height)
                    # make a bounding box for the tag
                    tag_box = BBox(x, y, x + tag_width, y + tag_height)
                    if not self._check_bndbox(bounding_boxes, tag_box):
                        aligned = True
                # print(f"Found spot for tag at {x}, {y}")
                bounding_boxes.append(tag_box)
                tags_to_write.append(Tag_BBox(tag_box, tag_id))
                image[tag_box.ymin:tag_box.ymax, tag_box.xmin:tag_box.xmax] = tag_image

            xmlstr = self._add_apriltag_to_xml(root, tags_to_write)

            if self.new_dir: # for testing, will make a new directory to save the images and xml files so the original data stays nice
                # print(f"Saving to new directory {self.new_dir_name}")
                # try to make test directory
                try:
                    os.mkdir(os.path.join(os.getcwd(), self.new_dir_name))
                except FileExistsError:
                    pass
                # update image_path and xml_file to save to the new directory
                image_path = os.path.join(self.new_dir_name, image_name)
                xml_file = os.path.join(self.new_dir_name, xml_file)
                # print(f"Saving to {image_path} and {xml_file}")
                # save the image and xml file, making the file if it doesn't exist
                cv2.imwrite(image_path, image)
                with open(xml_file, "w") as f:
                    f.write(xmlstr)
            else:
                # save the image and xml file, making the file if it doesn't exist
                with open(os.path.join(self.data_dir, xml_file), "w") as f:
                    f.write(xmlstr)
                # write the new image
                cv2.imwrite(image_path, image)    
            print(f"Saved to {image_path} and {xml_file}")

    def _add_apriltag_to_xml(self, root, tags):
        for tag in tags:
            # create a new object
            new_obj = ET.SubElement(root, "object")
            # create the name and id
            name = ET.SubElement(new_obj, "name")
            name.text = "april16h11_" + str(tag.id)
            id = ET.SubElement(new_obj, "tag_id")
            id.text = str(tag.id)
            # create the bndbox
            bndbox = ET.SubElement(new_obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(tag.BBox.xmin)
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(tag.BBox.ymin)
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(tag.BBox.xmax)
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(tag.BBox.ymax)
    
        # return the xml as a string
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        # make the string have only one line break per tag, not needed but looks better
        # ty copilot :)
        xmlstr = os.linesep.join([s for s in xmlstr.splitlines() if s.strip()])
        return xmlstr
    
    '''
    Checks if the superimposed apriltag bounding box overlaps with any of the existing bounding boxes
    bndbox: List of tuples of (BBoxs)
    tag_bndbox: BBox 
    returns: True if there is an overlap, False otherwise
    '''
    def _check_bndbox(self, bndbox, tag_bndbox):
        for box in bndbox:
            if (tag_bndbox.xmin < box.xmax and tag_bndbox.xmax > box.xmin and
                tag_bndbox.ymin < box.ymax and tag_bndbox.ymax > box.ymin):
                return True
        return False

img_dir = "/home/chris/tensorflow_workspace/2022Game/data/apriltag-imgs/tag16h5"
data_dir = "/home/chris/tensorflow_workspace/2022Game/data/videos"
trainer = ApriltagTrainer(img_dir, data_dir, new_dir=True)
trainer.augment()
trainer.generate_xml()
'''
# loop through all the xml files
for f in datadir:
    # get the name of the file without the extension
    name = f[:-4]
    print(name)
    # load xml file as a tree
    tree = ET.parse(f"data/{f}")
    # get the root of the tree
    root = tree.getroot()
    # add an object to the xml file
    obj = ET.SubElement(root, "object")
    # add the name of the object
    ET.SubElement(obj, "name").text = "TODO"
    # add the bounding box of the object
    bndbox = ET.SubElement(obj, "bndbox")
    ET.SubElement(bndbox, "xmin").text = "TODO"
    ET.SubElement(bndbox, "ymin").text = "TODO"
    ET.SubElement(bndbox, "xmax").text = "TODO"
    ET.SubElement(bndbox, "ymax").text = "TODO"

    # save the xml file
    # tree.write(f"data/{f}")
    # format xml before write
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    # make the string have only one line break per tag, not needed but looks better
    # ty copilot :)
    xmlstr = os.linesep.join([s for s in xmlstr.splitlines() if s.strip()])
    with open(f"test.xml", "w") as f:
        f.write(xmlstr)
    # tree.write("test.xml")
'''

