### Table of Contents

*   [Overview](#overview)
    
*   [Presentation on labeling images](#presentation-on-labeling-images)
    
*   [Initial Setup](#initial-setup)
    
*   [Finding Videos](#finding-videos)
    
*   [](#section)
    
*   [Labeling Images](#labeling-images)
    
    *   [List of object names](#list-of-object-names)
        
    *   [Per-object type best practices](#per-object-type-best-practices)
        
    *   [Hard Negative Images](#hard-negative-images)
        
*   [Creating Tensorflow TF Record files](#creating-tensorflow-tf-record-files)
    
*   [Training a model](#training-a-model)
    
*   [Exact Steps for Training a Model](#exact-steps-for-training-a-model)
    
*   [Training Outputs](#training-outputs)
    
    *   [Final Results](#final-results)
        
*   [Testing a model](#testing-a-model)
    
*   [More Documentation](#more-documentation)
    

This page outlines how to get started with tensorflow object detection

There are a few basic steps

*   Initial Setup (only needs to be done the first time after pulling a new container)
    
*   Finding and downloading videos which have the objects we care about
    
*   Extracting still images from those videos
    
*   Labeling the objects of interest in the extracted stills
    
*   Generating tensorflow input files
    
*   Training a model
    

Each will be explained in a section below

Overview
--------

But first, an introduction and outline to the entire process.

We're trying to write code to detect various objects in live video. To do this, we're using neural nets written using a framework from Google called Tensorflow.

Very basically, neural nets are mathematical constructs which (among other things) are good at finding patterns in data. In our case, we want them to both find where an object is located and what type of object it is. The neural net is fed a large amount of training data - in our case, labeled object locations in example images. From those examples, is uses various techniques to generalize from those examples into more generic ways to classify and localize objects in new images.

We're using a model called SSD, for Single Shot Multibox Detector. No, I'm not sure where the M from multibox went. Early object detection models used detection code separate from classification code - that is, it would have different steps for finding things in an image and then for figuring out what each object was. SSD and similar models instead combine the process into on single “shot” detecting multiple boxes, hence the name.

There are various ways to tune the detector. But the biggest initial part of the problem is getting good examples to train from. This typically has to be done by hand. After all, if we had an effective automatic way to detect these objects in images, we wouldn't need to create one. So much of the early work is going to be finding good example images and then labeling all of the objects we care about. In this context, labeling means using a specialized program to drag boxes around each object and then typing a label to go with it.

This training data is used to train the neural net. The network has a number of weights, which are tuneable parameters - basically coefficients for various mathematical functions. These constants start with random values. The training data is passed through the network and the result is compared against the known correct values given as part of the training data. The training process adjusts the weights to minimize the error on the training data. With any luck, this has created a solution general enough to also correctly identify new images.

Presentation on labeling images
-------------------------------

[https://docs.google.com/presentation/d/1DzvIlfS8eMRhz8rz0pd0_RoR_4SL-Y18PuZ6HOEJ6Ig/edit?usp=sharing](https://docs.google.com/presentation/d/1DzvIlfS8eMRhz8rz0pd0_RoR_4SL-Y18PuZ6HOEJ6Ig/edit?usp=sharing)

Initial Setup
-------------

This needs to be done to grab the container the first time you run.

```
git clone tensorflow_workspace
sudo true
tensorflow_workspace/setup.sh
tensorflow_workspace/docker-run
```

**Note** - this is a different container than the normal one used to e.g. build robot code. Be sure to run docker-run from the directory mentioned above.

This will pull the tensorflow container and start up a shell in it. Like the RobotCode, this docker-run script maps the tensorflow_workspace from outside the container into it. Files changed in that directory tree inside the container will be visible outside it, and vice-versa

Then, build the tool which extracts individual still frames from videos :

```
cd ~/tensorflow_workspace/tools
cmake .
make
```

Finding Videos
--------------

**Note** - for now there are already a lot of good quality images in our repo that need to be labeled. If you know of a good video we just can't live without, grab it. But for now, the priority should be labeling the ones we have.

Find a good video on youtube of the field. This will preferably be decent video quality, shot from around robot height, and obviously have the objects we care about in it.

In the URL, change youtube.com to youtu10.com, hit enter. It should take you youtube-saving video website with the video selected. Pick a high-res version and click download. Save the video to your local machine when prompted.

Copy/move the video to `tensorflow_workspace/2020Game/data/videos`.

Note that the file was likely downloaded from a browser running outside the container, which means the file was saved outside the container as well. Thus, you'll need to copy it using a shell script or file explorer window outside the container. Since the tensorflow_workspace is visible both inside and outside the container, copying it will make it available for subsequent steps run inside the container.

Go into a shell running inside the container. Change to the directory in tensorflow_workspace with the video :

```
cd ~/tensorflow_workspace/2020Game/data/videos
extract_frame <video_name\>
```

That will create a number of PNG image files from various frames in the video. The program tries to find stills which are the clearest, but also tries to space the saved images so they're not too close to each other in the video.

Labeling Images
---------------

The goal here is to draw a box around each instance of an object we are trying to detect

#### Reserve a set of images to work on

To prevent duplication of effort, there's a shared google sheet here : [https://docs.google.com/spreadsheets/d/1JPGZgy09rbxBKXkN_m9I6FntS7K6kI0pVfRcza1Aans/edit](https://docs.google.com/spreadsheets/d/1JPGZgy09rbxBKXkN_m9I6FntS7K6kI0pVfRcza1Aans/edit)

Use that to indicate you're working on a set of images. Mark them as in-progress with your name or something. This way people won't waste time labeling the same sets of images

#### How to label

The list of types is in

`tensorflow_workspace/predefined_classes.txt`

The names used to label images must match one of those exactly. If you find another useful image type, create a task for it - it will need to be added to both `tensorflow_workspace/predefined_classes.txt` and `2020Game/data/2020_label_map.pbtxt`.

After running `labelImg`, you need to select a directory which contains a set of images to label. The keyboard shortcut for this is CTRL-U, then select the `2020GameData/data/video` directory. This will load all the images from that directory into labelImg's list of images - see the lower right pane of the GUI. In this list, to the first image from the video you want to label and click on it.

This will bring up the image itself in the labelImg GUI.

Type w to draw a rectangle around an object. Type the name (or select it from the list) of the object. Note the list should be pre-populated with valid label names, so after typing the first few letters it should narrow down the options.

Repeat until everything in the current image is is labeled.

The labelImg website has additional hot-keys which can be used [here](https://github.com/tzutalin/labelImg "https://github.com/tzutalin/labelImg").

For objects which are partially blocked, use your judgement on whether or not there's enough there to detect

When finished with an image, hit space to save it (accepting the default file name is fine). This will save the drawn rectangles to an xml file - the name of the file is just the png image file name with png replaced by xml.

a and d scroll between images

In many cases, frames will be very similar. Two options

1.  copy the xml from one frame to the next one and edit it if needed. This copies the marked detection from one image onto another. This can serve as a quick starting point for cases where, for example, the camera is fixed. Stationary field elements will be in the same place, so most of the objects won't have moved. The detection rectangles for moving objects can then be updated in the new image (drag/resize the windows) and then saved.
    
2.  delete the duplicate images and skip them. This is best if, say, the video is a still image. Adding additional identical detection doesn't help that much. In this case, you'll probably want to reload the list of files in labelImg after deleting the duplicate files - this way the program knows not to try to scroll to them when moving between images
    

Images which have no objects in them can be deleted

Once all the image files are labeled, add the video, images and xml files to git and commit them.

### List of object names

*   power_cell
    
*   red_power_port_high_goal
    
*   blue_power_port_high_goal
    
*   red_power_port_low_goal
    
*   blue_power_port_low_goal
    
*   power_port_yellow_graphics
    
*   red_power_port_first_logo
    
*   blue_power_port_first_logo
    
*   red_loading_bay_tape
    
*   blue_loading_bay_tape
    
*   red_loading_bay_left_graphics
    
*   red_loading_bay_right_graphics
    
*   blue_loading_bay_left_graphics
    
*   blue_loading_bay_right_graphics
    
*   red_tape_corner
    
*   blue_tape_corner
    
*   red_ds_light
    
*   blue_ds_light
    
*   ds_light
    
*   control_panel_light
    
*   yellow_control_panel_light
    
*   shield_generator_light
    
*   red_shield_generator_light
    
*   blue_shield_generator_light
    
*   shield_generator_backstop
    
*   shield_generator_first_logo
    
*   shield_generator_yellow_stripe
    
*   shield_generator_floor_center_intersection
    
*   red_black_shield_generator_floor_intersection
    
*   blue_black_shield_generator_floor_intersection
    
*   red_blue_black_shield_generator_floor_intersection
    
*   red_shield_pillar_intersection
    
*   blue_shield_pillar_intersection
    
*   ds_numbers
    
*   control_panel
    
*   red_robot
    
*   blue_robot
    

### Per-object type best practices

In general, try and outline as closely as possible just the object itself. There's naturally going to be some background data saved because few objects are a perfect rectangle when viewed from any image. But do your best not to add too much extra border - try to be consistent. Note that in most cases due to the angle of the image or the position of it the rectangle won't be lined up exactly even with square objects. That's OK - just make sure to have the edges of the bounding box come as close as possible to touching the corners of the object.

Given all the metal and plastic, there will be reflections of objects in various images. I'm currently thinking we shouldn't label these. My rationale is that we don't want them to be detected as if they're real, and labeling just encourages the AI to detect them. Honestly, it will likely still pick them up as objects, but not labeling them at least gives us a chance to distinguish between real and reflected images.

#### Game piece

This is pretty straighforward. Try to outline the ball exactly. It will be a judgement call on how much overlap between balls before not marking the one in back.

#### Power Port

Targets for the high and low goal, along with random graphics which might be useful for localization. Most of these will have different labels for red vs. blue. The exception is the yellow graphics, since there's no background color to work with near it.

For the high goal, outline the entire hexagon. For the low goal, outline the entire red hexagon. For the yellow graphics, outline the darker grey box (with the weird cut-off corner) containing the yellow graphics and text. For the first logo, draw a box which contains the letters FIRST and the FIRST triangle-circle-square rectangle above it. There will be a good bit of background color included.

#### Loading Bay

Like the power port, the loading bay will have blue and red versions depending on the area being labeled.

The right graphics object is the set of letters on a colored background on the right side. Outline the entire colored area The left graphics object is the corresponding one on the left. Here, make a similar rectangle. Make it as wide as most of the object - don't worry that it will include a bit of grey and cut off a bit of blue near the top where it is angled. Just extend the rectangle up to the driver station glass. The loading_bay_tape object should be the rounded rectangle graphics holding the retro tape, plus the block of color to the right of that. Distinguishing between red and blue might be useful for localization, so we'll include the color graphics in the object.

#### Tape corner

This is an experiment.

The idea is to mark any right angle-ish corner made out of colored tape on the field floor. The places this can be seen are

*   Around the trench run
    
*   The apex of the triangles near the loading bay & power port
    

Try to grab enough of a rectangle to see the corner clearly against the gray carpet background. Given the weird angles that the corners will be viewed at depending on the camera angle, exactly how much to mark will be a judgement call. Look at what others have done and do your best. We can adjust as we learn more about the best way to use these.

#### Shield Generator

This is a busy area of the field. I've broken this up into sections for stuff off the ground and stuff on the ground.

##### High Shield Generator

There are a few sets of graphics on the top of the shield generator :

*   The first logo is black-on-gray FIRST letters and logo. Like the logo on the power port, outline the a box containing the letters and the logo. It will naturally grab a good bit of grey background as well.
    
*   The yellow stripe is the yellow stripe graphics at the corner of the structure. Make a rectangle where two corners hit the endpoints of the yellow graphics
    

The backstop is the big metal piece that the hangers will bump into if they lean too much. Try to make the box include the full width of the bottom part of it, and go high enough to include the full length of the larger bolts holding it together.

##### Shield Generator Floor

This is labeling various markings on the floor. Like the tape corners described above, finding exactly the right amount of floor to include along with these corners is going to be an experiment.

The red_black_shield_generator_floor_intersection is where the red and black raised bits on the floor intersect. Try to mark enough to see red on both sides of the black bar. Same with the blue_black_shield_generator_floor_intersection object, just for the blue bar instead.

The red_blue_black_shield_generator_floor_intersection is where all 3 colored bars come together on the floor. Try to make the rectangle big enough that all three colors are contained in it.

shield_generator_floor_center_intersection is the floor at the center of the shield generator where the black bars cross each other.

The red_shield_pillar_intersection and blue_shield_pillar_intersection are where the metal supports for the shield generator structure hit the floor. There's a darker platform they're placed on along with the some red or blue markings which line up with the raised bits on the floor. Try to label enough to capture all of that colored marking against the dark background. From most angles this will also capture a good bit of lighter colored carpet - that's OK.

#### Driver station wall and assorted status lights

The blue and red lights are the stack of colored lights above each driver station. There's also a ds_light object for videos taken where the lights are turned off. This should never happen in a match, but it might be useful before the match - letting us use even the unlit lights to figure out our location on the field.

The ds_numbers object is either the 7-segment boxes team number or time near the top of the driver station windows. Note that since the numbers will be variable length, we should try to outline the entire “number box”, for lack of a better term. That is, the full dark box where numbers could show up.

The control panel also has a status light. Mark this. Like the ds lights, there is a label for the unlit version (control_panel_light) as well as one for when the light is lit (yellow_control_panel_light).

Finally, there are status lights on the shield generator. Like the ds lights, there are separate labels for when the lights are light (blue_shield_generator_light and red_shield_generator_light) and one for the unlit version (shield_generator_light).

### Hard Negative Images

We might be able to improve the net's ability to reject false positives by providing images which contain none of the object types we're looking for. This would allow the training process to identify cases where it is detecting things it shouldn't - by definition, anything detected in an image with no valid objects is a false positive.

It looks like saving on an image with nothing labeled in TF adds it as a hard negative image. So we just need a collection of images with nothing field-related in them and use labelImg to save them with an empty set of labels.

TODO - might be nice to have a script which auto-generates a set of XML label files with no images in them for a given set of images?

Creating Tensorflow TF Record files
-----------------------------------

The files used to train tensorflow models need to be put into so-called TFRecord files. The script to do this is

 python /home/ubuntu/tensorflow_workspace/2020Game/data/create_tf_record.py \\
        --label_map_path=/home/ubuntu/tensorflow_workspace/2020Game/data/2020Game_label_map.pbtxt \\
        --data_dir=/home/ubuntu/tensorflow_workspace/2020Game/data/videos \\
        --alt-data_dir=/home/ubuntu/tensorflow_workspace/2019Game/data/videos \\
        --output_dir=/home/ubuntu/tensorflow_workspace/2020Game/data

This will read the image files in the 2020Game/data/videos directory along with their corresponding .xml files. It also grabs some older field images from the 2019Game directory, used for images with labeled objects which don't change from year to year - mainly things like the driver station numbers and so on. Using this data, it will create two sets of TFRecord files, each holding image data used to train the models. These TFRecord files just combine a bunch of images along with their labels into a few bigger files for efficiency. Typically reading thousands of small files is slower than a reading a few larger ones, even when the overall amount of data read is the same.

When training the model, a certain amount of labeled image data is reserved for validation. That validation data is not used to train the model at all and is only used after the model is trained to test it. This provides an independent set of data to make sure the training is not learning just the actual training data but is in fact actually _generalizing_ - learning to detect whole classes of objects rather than just the exact specific objects given in the training data set.

So the two sets of TFRecord files are

*   1 larger (as in more numerous, not like image height and width) set of images for training the model
    
*   1 smaller set of independent images for validating the model.
    

Training a model
----------------

A tensorflow model describes the configuration of a particular neural network. This model describes the number of layers which make up the model and the particulars of each layer. The TF object detection toolbox we're using abstracts this process a bit - instead of describing the exact detail of each layer, a config file is used to specify the important parts of the model and the toolbox handles the rest.

Models are set up using config files. These config files are (will be) in tensorflow_workspace/2020Game/models/model.

The neural net itself is a large mathematical equation which, given an input (an image in our case), produces various numeric output. For us, the output is the location and type of object detected. The model contains a (huge) number of weights - constants which are part of those equations.

The training process uses labeled training data to generate values for those constants. It gradually refines the constants so that the equations of the network produce output which matches the labels.

To train a model with the data created in the previous step, run train.sh in the 2020Game/models subdir.

This will be much faster on a system with a GPU installed. Details are TBD on how to get students set up to run in that configuration.

Exact Steps for Training a Model
--------------------------------

Use the workstation for this, since it has a lot of memory and a powerful GPU.

If not in docker, run

~/tensorflow_workspace/docker-run-gpu

to start a container.

Then make sure the repo has been updated with the latest labeled images and other changes :

cd ~/tensorflow_workspace
git checkout master
git pull \-r

Then create TF records for training using the command from the Creating Tensorflow TF Record files section above

Then, actually start the training. This is done by creating a directory for the new model and running from it.

cd ~/tensorflow_workspace/2020Game/models
mkdir random_model_name
cd random_model_name

Obviously, use a more descriptive name.

In ~/tensorflow_workspace/2020Game/models there is an example script to train a generic model, called train.sh. Copy this file to the model directory you've created.

cp ../train.sh .

Edit the file so that MODEL_DIR is the full pathname of the new training directory you created.

Next up, you'll need a config file. This file describes in detail the model you're about to train along with the location of the data to train it. Several pre-configured model files are located in ~/tensorflow_workspace/2020Game/models. The one we are currently using is called ssd_mobilenet_v2_512x512_coco.config. Breaking that down :

*   SSD is the object detection model, and mobilenetv2 is the object classifier used by SSD
    
*   512×512 is the resolution input images are scaled to when input to the net
    
*   CoCo is the nickname for a primitive 8-bit home computer from the 1980s - [https://en.wikipedia.org/wiki/TRS-80_Color_Computer](https://en.wikipedia.org/wiki/TRS-80_Color_Computer "https://en.wikipedia.org/wiki/TRS-80_Color_Computer")
    
*   No, actually COCO is a “Common Objects in Context” - a database of images used to testing machine learning models. In our case, the mobilenet V2 object classifier is pre-trained on these objects. We then use [https://builtin.com/data-science/transfer-learning](https://builtin.com/data-science/transfer-learning "https://builtin.com/data-science/transfer-learning") to fine tune the model trained on generic objects so it can identify our very specific set of FRC field objects
    

All of the config is important, but we'll avoid changing much of it for now. The things we might have to change

*   num_classes - the number of different categories of objects that the trained data has labeled. If we add new types of objects, this value will have to change to match.
    
*   input_path under train_input_reader and eval_input_reader needs to point to the TF record files generated in the the Creating Tensorflow TF Record files section above
    
*   fine_tune_checkpoint needs to point to a pretrained model ckpt (checkpoint file). This checkpoint is the saved version of the AI model trained on the COCO (or other) dataset mentioned above. Typically this will be in the same location from run to run, but just in case it can been configure here.
    

Typically we've been copying the config file to the training directory even if nothing in it changes. That way the config needed to reproduce a run is all contained in one place.

In train.sh, make sure PIPELINE_CONFIG_PATH points to the config file you want to use.

Then, to start training, run ./train.sh. Get ready to wait, this takes a while - probably a day or two.

Training Outputs
----------------

The training process produces a number of output files.

A number of checkpoint files are produced, typically called model.ckpt.<stuff>. These periodically save the state of training so it can be resumed if interrupted.

There's a directory called best which saves copied of the checkpoint files from the best models evaluated during training. Typically we'd expect the ones from the end of training to be best, but this isn't always true so it is nice to have the best results backed up if needed.

Several events files are saved. These record logs of data during training - for example, we can go back and look at training rates, model loss values and accuracy vs. training iteration number. That's useful for debugging and tuning models. The tool for visualizing the data in these files is called [https://www.tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard).

To run it, switch to the directory where the traning was run (look for checkpoint files and an eval directory). Run `python3 /usr/local/bin/tensorboard --log .`.

### Final Results

Once training is complete, we'll have a number of ckpt files representing the best model weights found during the training process. These are not directly usable by our testing tools. Instead, the ckpy files need to be converted into something called “frozen graphs”, which for reasons beyond the scope of the document to explain typically have a .pb extension. Luckily we do have scripts to do this conversion.

The script we currently used is called create_trt_graph.py. This actually does 2 things. First, it takes a checkpoint file and converts it to a frozen graph. Next, it takes the graph and runs it through an optimization step using a tool called TensorRT (hence the trt in the name).

Using this script requires editing a few variables at the top of the file. FROZEN_GRAPH_NAME should be changed to something descriptive, probably related to the name of the training directory. The SAVED_MODEL_DIR and CHECKPOINT_NUMBER variables should match the checkpoint file output during training (MODEL_CHECKPOINT_PREFIX typically doesn't change but pay attention just in case). Finally, the CONFIG_FILE name should also match the config file used to train the model.

Running this script will produce a FROZEN_GRAPH_NAME.pb output which is the unoptimized frozen graph along with a trt_FROZEN_GRAPH_NAME.pb file which is the optimized counterpart of it. Pay attention to the full path names of these variables to help you find them.

This script is another possible TODO to make all of the above editing work using command line arguments instead.

Testing a model
---------------

Next up, we have a script which loads a model from a .pb frozen graph file and runs it on videos or images. This is the test_detection.py script.

Edit the MODEL_NAME variable to match the directory specified in the --output_directory argument.

Edit the cap variable to point to a video to use as input. A number are included, simply uncomment one that looks interesting.

Run

python test_detection.py

To test against a directory full of images, comment out the while(True): loop starting just after the list of videos. Uncomment the block below it, starting with TEST_IMAGE_PATHS.

Change TEST_IMAGE_PATHS to point to the list of images to test in this mode.

Yes, having to edit the test_detection.py script to try out different videos or trained models is terrible. A good TODO would be to make those options command line arguments for the script.

More Documentation
------------------

*   Add docs for TRT UFF code
    
*   What to modify in config files - learning rate / decay, loss function, input size, more advanced like a new backbone, etc
    
*   Tensorboard to monitor results, more detail needed
    
*   How to save dir after training - need a process for this, what to save in the git repo, naming conventions, etc
    
*   Moving frozen_graph.pb to robot code repo

[Home](/README.md)