#TODO list
# inserted apriltags can overlap themselves
# tighter bounding boxes around resized(?) tags
# occasional images have apriltags inserted but bad labels
# port to pascal-voc (will save a bunch of code)
# Simplify - can we run augmentation in the code which overlays them on images rather than pregenerating them?
# Maybe also simplify - just pick a random image from the source dir, overlay a random number of tags, and continue until the desired # of new images or augmented tags are created

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
import numpy as np
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
    def __init__(self, tag_path, data_dir, new_dir=False, logging=True, prefix=None) -> None:
        # yes i know the logging module is builtin and exists
        self.log = logging
        self.prefix = prefix
        self.aprildirlist = None
        self.new_dir = new_dir
        self.new_dir_name = "data/test"
        self.tags_to_add = [1, 1, 1, 1, 2, 2, 2, 3, 3] # ratio of how many tags to add to each image
        self.data_dir = data_dir
        self.tag_path = tag_path
        if self.log:
            print(f"Tag path: {tag_path} \nData path: {data_dir}")
        print(f"Initializing Augmentor with tag path {self.tag_path}")
        self.p = Augmentor.Pipeline(self.tag_path)    
        # do custom rotaions
        #self.p.rotate(probability=0.7, max_left_rotation=5, max_right_rotation=5) 
        #self.p.rotate_random_90(probability=0.75)
        #self.p.zoom(probability=0.5, min_factor=0.7, max_factor=1.4)
        self.p.skew(probability=0.5, magnitude=0.3)
        #self.p.random_distortion(probability=0.5, grid_width=5, grid_height=5, magnitude=3)
        self.p.random_brightness(probability=0.3, min_factor=0.5, max_factor=1.5)
    
    def _zoom_at(img, zoom, coord=None):
        """
        Simple image zooming without boundary checking.
        Centered at "coord", if given, else the image center.
        img: numpy.ndarray of shape (h,w,:)
        zoom: float
        coord: (float, float)
        """
        # Translate to zoomed coordinates
        h, w, _ = [ zoom * i for i in img.shape ]
        
        if coord is None: cx, cy = w/2, h/2
        else: cx, cy = [ zoom*c for c in coord ]
        
        img = cv2.resize( img, (0, 0), fx=zoom, fy=zoom)
        img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
                int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
                : ]
        
        return img
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
        # read all the images output by the pipeline and randomly resize them and write them again to the test directory
        # this is to make the images more realistic
        # get a list of all the apriltag images
        self.aprildirlist = os.listdir(os.path.join(self.tag_path, "output"))
        # read in all the images
        for image in self.aprildirlist:
            img = cv2.imread(os.path.join(self.tag_path, "output", image))

            # resize the image randomly between a quarter and double the size
            scale = random.uniform(0.2, 1.9)
            img = cv2.resize(img, (0,0), fx=scale, fy=scale)
            # rotate image random between 0 and 360 degrees and have the new space be white
            rows, cols, _ = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), random.uniform(-25, 25), 1)
            # make the new space white
            img = cv2.warpAffine(img, M, (cols, rows), borderValue=(255, 255, 255))

            # show before and after
            #cv2.imshow("before", img)
            #cv2.imshow("after", rotate_img)
            #cv2.waitKey(4000)
            # write the image
            cv2.imwrite(os.path.join(self.tag_path, "output", image), img)
    
    def postprocess_imgs(self):
        # get a list of all final images
        listdir = os.listdir(self.new_dir_name)
        # remove all non png files
        listdir = [x for x in listdir if x[-4:] == ".png"]
        print(f"listdir: {listdir}")
        # apply directional blur to simulate motion blur
        for image in listdir:
            print(f"image: {image}")
            # apply to only 5% of images
            if random.randint(0, 100) < 5:
                img = cv2.imread(os.path.join(self.new_dir_name, image))
                # apply directional blur
                psf = np.zeros((50, 50, 3))
                psf = cv2.ellipse(psf, 
                                (25, 25), # center
                                (22, 0), # axes -- 22 for blur length, 0 for thin PSF 
                                15, # angle of motion in degrees
                                0, 360, # ful ellipse, not an arc
                                (1, 1, 1), # white color
                                thickness=-1) # filled

                psf /= psf[:,:,0].sum() # normalize by sum of one channel 
                                        # since channels are processed independently

                imfilt = cv2.filter2D(img, -1, psf)
                # write the image
                cv2.imwrite(os.path.join(self.new_dir_name, image), img)
                # show before and after
                #cv2.imshow("before", img)
                #cv2.imshow("after", imfilt)
                #cv2.waitKey(0)
    
    '''
    Usage: Uses the augmented images from tag_path/output and goes through each xml file in the 
    data_dir and the corresponding image. It then adds the apriltag to the image and modifies the
    xml file to reflect the new bounding box coordinates. 
    It then saves the new image and xml file in the data_dir, _overwriting_.
    '''
    def generate_xml(self, dir=None):
        parsedpngs = []
        # get a list of all the apriltag images
        self.aprildirlist = os.listdir(os.path.join(self.tag_path, "output"))
        if self.log:
            print(f"self.aprildirlist: {self.aprildirlist}")
        #time.sleep(3)
        # shuffle list
        random.shuffle(self.aprildirlist)

        if dir:
            datadirlist = os.listdir(dir)
        else:
            dir = self.data_dir
        datadirlist = os.listdir(dir)
        print(f"datadirlist: {datadirlist}")
        time.sleep(3)
        datadirlist = [x for x in datadirlist if x[-4:] == ".xml"]
        for xml_file in datadirlist:
            if self.log:
                print("\n\n\n")
            # print(f"Processing {xml_file}")
            # get the image name from the xml file data
            tree = ET.parse(os.path.join(dir, xml_file))
            root = tree.getroot()
            image_name = root.find('filename').text
            # print(f"Processing {image_name}")
            image_path = os.path.join(dir, image_name)
            parsedpngs.append(image_name)
            # read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error reading {image_path}, skipping")
                continue
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
            if self.log:
                print(f"Number of tags to add: {to_add}")
            
            for _ in range(to_add):
                if len(self.aprildirlist) == 0:
                    print("Out of images from this dir")
                    break
                apriltag = self.aprildirlist.pop()
                tag_path = os.path.join(self.tag_path, "output", apriltag)
                # read an augemented apriltag image
                tag_image = cv2.imread(tag_path)
                if self.log:
                    print(f"Image path: {tag_path}")
                # get tag id, 5 characters before the . in the first instance of .png
                tag_id = int(apriltag[apriltag.find(".png")-5:apriltag.find(".png")])
                # get the tag dimensions
                tag_height, tag_width, _ = tag_image.shape
                if self.log:
                    print(f"Tag id: {tag_id}, file name {apriltag} \n Tag height: {tag_height}\n Tag width: {tag_width}")
                aligned = False
                cnt = 0
                # print("Trying to find a good spot for the tag")
                while not aligned:
                    if cnt > 100:
                        print("Couldn't find a good spot for the tag, skipping")
                        break
                    # get a random x and y coordinate for the tag
                    x = random.randint(0, width - tag_width)
                    y = random.randint(0, height - tag_height)
                    # make a bounding box for the tag
                    tag_box = BBox(x, y, x + tag_width, y + tag_height)
                    if not self._check_bndbox(bounding_boxes, tag_box):
                        aligned = True
                    cnt += 1
                # print(f"Found spot for tag at {x}, {y}")
                bounding_boxes.append(tag_box)
                tags_to_write.append(Tag_BBox(tag_box, tag_id))
                image[tag_box.ymin:tag_box.ymax, tag_box.xmin:tag_box.xmax] = tag_image
            # replace image name with the new image name + prefix
            if self.prefix:
                image_name = self.prefix + image_name
                xml_file = self.prefix + xml_file
                # set to root
                root.find('filename').text = image_name
                # replace image path with the new image path + prefix
                image_path = os.path.join(dir, image_name)
                
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
                with open(os.path.join(dir, xml_file), "w") as f:
                    f.write(xmlstr)
                # write the new image
                cv2.imwrite(image_path, image)    
            if self.log:
                print(f"Saved to {image_path} and {xml_file}")

    '''
    Takes in a directory, dir, and goes through each png file in the directory. It then adds the apriltags 
    to the image and saves the new image in the same directory, overwriting the old image. It also makes 
    a NEW xml file for each image with the bounding box coordinates of the apriltags.
    '''
    def generate_from_png(self, dir):
        # get a list of all the apriltag images
        if not self.aprildirlist:
            self.aprildirlist = os.listdir(os.path.join(self.tag_path, "output"))
            random.shuffle(self.aprildirlist)
        
        # get a list of all the png images in the directory
        alllist = os.listdir(dir)
        print(f"alllist: {alllist}")
        # remove all xml and png corresponding to the xml files
        for xml_file in list(alllist):
            if xml_file[-4:] == ".xml":
                # read the file to get the image name
                # just add .png to find the image
                # remove the image and xml file from the list

                tree = ET.parse(os.path.join(dir, xml_file))
                root = tree.getroot()
                image_name = root.find('filename').text
                # remove the image and xml file from the list
                print(image_name)
                print(xml_file)
                try:
                    alllist.remove(image_name)
                except:
                    print(f"Couldn't remove {image_name}")
                    time.sleep(0)
                try:
                    alllist.remove(xml_file)
                except:
                    print(f"Couldn't remove {xml_file}")
                    time.sleep(0)

        pnglist = [x for x in alllist if x[-4:] == ".png"]
        for png in pnglist:
            print(f"png: {png}")
            # read the image
            image = cv2.imread(os.path.join(dir, png))
            # get the image dimensions
            height, width, _ = image.shape
            bounding_boxes = [] # just for checking tag collision here
            tags_to_write = []
            to_add = random.choice(self.tags_to_add)
            if self.log:
                print(f"Number of tags to add: {to_add}")
            # add the tags to the image
            for _ in range(to_add):
                apriltag = self.aprildirlist.pop()
                tag_path = os.path.join(self.tag_path, "output", apriltag)
                # read an augemented apriltag image
                tag_image = cv2.imread(tag_path)
                if self.log:
                    print(f"Image path: {tag_path}")
                # get tag id, 5 characters before the . in the first instance of .png
                tag_id = int(apriltag[apriltag.find(".png")-5:apriltag.find(".png")])
                # get the tag dimensions
                tag_height, tag_width, _ = tag_image.shape
                if self.log:
                    print(f"Tag id: {tag_id}, file name {apriltag} \n Tag height: {tag_height}\n Tag width: {tag_width}")
                aligned = False
                # print("Trying to find a good spot for the tag")
                cnt = 0
                # print("Trying to find a good spot for the tag")
                while not aligned:
                    if cnt > 100:
                        print("Couldn't find a good spot for the tag, skipping")
                        break
                    # get a random x and y coordinate for the tag
                    x = random.randint(0, width - tag_width)
                    y = random.randint(0, height - tag_height)
                    # make a bounding box for the tag
                    tag_box = BBox(x, y, x + tag_width, y + tag_height)
                    if not self._check_bndbox(bounding_boxes, tag_box):
                        aligned = True
                    cnt += 1

                # print(f"Found spot for tag at {x}, {y}")
                bounding_boxes.append(tag_box)
                tags_to_write.append(Tag_BBox(tag_box, tag_id))
                image[tag_box.ymin:tag_box.ymax, tag_box.xmin:tag_box.xmax] = tag_image
            # save the image and make a corresponding xml file
            if self.prefix:
                print(f"Prefix: {self.prefix}")
                png = self.prefix + png
            cv2.imwrite(os.path.join(self.new_dir_name, png), image)
            xmlstr = self._make_xml(png, tags_to_write, dir, image)
            with open(os.path.join(self.new_dir_name, png[:-4] + ".xml"), "w") as f:
                f.write(xmlstr)
            if self.log:
                print(f"Saved to {png} and {png[:-4] + '.xml'}")

    '''
    Makes a fully new xml file, adds the tags to it, returns string to be written to file
    '''
    def _make_xml(self, png, tags_to_write, dir, img):
        # make a new xml file for the image
        xml_file = png[:-4] + ".xml"
        # make a new root element
        root = ET.Element("annotation")
        # make the folder element
        folder = ET.SubElement(root, "folder")
        folder.text = "images"
        # make the filename element
        filename = ET.SubElement(root, "filename")
        filename.text = png
        # make the path element
        path = ET.SubElement(root, "path")
        path.text = os.path.join(dir, png)
        # make the source element
        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"
        # make the size element
        size = ET.SubElement(root, "size")
        width = ET.SubElement(size, "width")
        height = ET.SubElement(size, "height")
        depth = ET.SubElement(size, "depth")
        # set the size element values
        height.text = str(img.shape[0])
        width.text = str(img.shape[1])
        depth.text = str(img.shape[2])
        # make the segmented element
        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"
        # make the object elements
        for tag in tags_to_write:
            # make the object element
            object = ET.SubElement(root, "object")
            name = ET.SubElement(object, "name")
            name.text = f"april_tag_{tag.id}"
            pose = ET.SubElement(object, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(object, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(object, "difficult")
            difficult.text = "0"
            bndbox = ET.SubElement(object, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(tag.BBox.xmin)
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(tag.BBox.ymin)
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(tag.BBox.xmax)
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(tag.BBox.ymax)
            id = ET.SubElement(object, "id")
            id.text = str(tag.id)
        # make the xml string
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
        xmlstr = os.linesep.join([s for s in xmlstr.splitlines() if s.strip()])
        return xmlstr

    def _add_apriltag_to_xml(self, root, tags):
        for tag in tags:
            # create a new object
            new_obj = ET.SubElement(root, "object")
            # create the name and id
            name = ET.SubElement(new_obj, "name")
            name.text = f"april_tag_{tag.id}"
            id = ET.SubElement(new_obj, "tag_id")
            id.text = str(tag.id)
            # add fields for difficult and truncated and pose (not used)
            difficult = ET.SubElement(new_obj, "difficult")
            difficult.text = "0"
            truncated = ET.SubElement(new_obj, "truncated")
            truncated.text = "0"
            pose = ET.SubElement(new_obj, "pose")
            pose.text = "Unspecified"
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

img_dir = "/home/ubuntu/tensorflow_workspace/2022Game/data/scaled"
data_dir = "/home/ubuntu/tensorflow_workspace/2022Game/data/videos"
trainer = ApriltagTrainer(img_dir, data_dir, new_dir=True, logging=False)
trainer.augment(n=10000)
print("Done augmenting")
trainer.generate_xml()

import timeit
# print("Time to run: ", timeit.timeit("trainer.generate_xml()", setup="from __main__ import trainer", number=1))
print('Running on 2023 data')
img_dir = "/home/ubuntu/tensorflow_workspace/2023Game/data/videos"

trainer.generate_xml(dir=img_dir)
trainer.generate_from_png(img_dir)
trainer.prefix = "round_2" # lots more images
trainer.generate_xml(dir="/home/ubuntu/tensorflow_workspace/2023Game/data/combined_88_test")
trainer.generate_from_png(img_dir)
#trainer.prefix = "round_3" # lots more images
#trainer.generate_xml(dir=img_dir)
#trainer.generate_from_png(img_dir) 

trainer.postprocess_imgs()

prefix = "/home/ubuntu/tensorflow_workspace/2023Game/data/test"

def chris_to_ubuntu(dir):
    # change all instances of /home/chris to /home/ubuntu in the xml files
    for file in os.listdir(dir):
        if file.endswith(".xml"):
            # parse the xml file
            tree = ET.parse(os.path.join(dir, file))
            root = tree.getroot()
            # get the path to the image
            path = root.find("path")
            # get image name from xml field filename
            filename = root.find("filename")
            # replace path with prefix + filename
            path.text = os.path.join(prefix, filename.text)
            # write the new xml file
            tree.write(os.path.join(dir, file))


#chris_to_ubuntu("/home/ubuntu/tensorflow_workspace/2023Game/data/test")

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
