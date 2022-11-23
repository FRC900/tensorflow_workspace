# resize all images in a directory to a given size

import os
import cv2

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

# scale all images in a directory to a given size using a scaling factor
def scale_all_images_in_dir(dir, scale):
    for filename in os.listdir(dir):
        if filename.endswith(".png"):
            img = cv2.imread(dir + "/" + filename)
            img = image_resize(img, width = int(img.shape[1] * scale))
            # save image as png
            try:
                os.mkdir(os.getcwd() + "/scaled")
            except:
                pass
            # write images to scaled directory
            cv2.imwrite(os.getcwd() + "/scaled/" + filename, img)
scale_all_images_in_dir("apriltag-imgs/tag16h5", 16)