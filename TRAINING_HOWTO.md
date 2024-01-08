
The tl;dr :

Everything here can be run from the main robot docker container. 

Training scripts are moved to the robot code repo, in zebROS\_ws/src/tf\_object\_detection/src.  
The main script to use is called train.py.  This takes a number of command line ags, but the defaults are set up to train using 2023 data.  Run this script from the 2023Game subdir in this repo.

The output will end up in runs/detect/train\*, with the number after train being automatically incremented each time training is run.  The outputs will be in the weights subdir.  

Once the new trained models have been checked out (use cpu-infer-video.py or infer-video.py) the .pt, .onnx and calib\*.bin files shoud be copied over into the tf\_object\_detection/src directory and git added/committed/merged.

Updating to new years

Create a new <Year>Game subdir, and under it a data/videos one.  Add videos, extract still images from them using tools/extract\_frames.  Data is labeled using the labelImg tool.  See the Videos and Labeling Images section here : [tensorflow_intro_and_labeling.md](tensorflow_intro_and_labeling.md)

Don't forget to run the script to remove the last year's labels from the dataset. These sorts of scripts are now in tensorflow\_repo/scripts

The model config is in zebROS\_ws/src/tf\_object\_detection/src/FRC2023.yaml.  This has the labels for that year's objects, and also hard-codes the post-processed dataset directory to a specific year's directory.  
config\_frc2023.py references the FRC2023.yaml file, so there should also likely be a new version of that. Although a good TODO might be to turn the code in there into a class which takes a .yaml config file as an argument so we don't have to edit it every year.  The config .yaml itself is an input to the training script, so that could be passed into a generic year-agnostic config to get the OBJECT\_CLASSES and COLORS.

Probably also makes sense to update all of the default command line args for the various scripts to point to the new year's info.

General links

Current model is YOLOv8, so their web site will have tons of docs much more up to date than anything we can generate : https://docs.ultralytics.com/

There's also a more detailed description of the training script operations here : https://docs.google.com/document/d/1h0AIUCuoX\_-24Szat5BqxkQ2ultraBK6oCE4nSzPiw0/edit?usp=sharing (note - the \ before the _ should be removed if copy-pasting from this as a raw text file). This was more of a planning doc than a post-mortem, so the script likely has improvements. But it should be a decent overview of the theory of operation.
