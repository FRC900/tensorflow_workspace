
Training requires a few things

A config for the model being trained
A script to run the training
A place to put output files

Config prototypes are in the model dir. They're named more or less appropriately for what they are supposed to do.
The script for training is train.sh and the like. Most of the copies simply change which config file is used, otherwise they're all the same
A place to put output files is best served by creating a new directory to train in.  Make a copy of both the config and training script in that directory and run from there. Edit the train.sh script to point to the correct config.  Update MODEL\_DIR to point to the new directory as well to insure the output files end up there.  Having the config, train .sh script and output files in that one directory will make it easier to figure out what's what after training a number of slightly different models.

Config File Walkthrough

There's a lot of data in the config file, much of it poorly documented.
Best approach is to start with a predefined config - look in models/research/object\_detection/samples/config for examples to use as a basis.  The models are documented (a bit) in  ~/models/research/object\_detection/g3doc/tf1\_detection\_zoo.md

There are a few things to modify to get these configs to work with our data.
First off, change the number of classes to match the number of objects we have defined, plus 1.  Find the list of objects in tensorflow\_workspace\2020Game\data\2020Game\_label\_map.pb. The extra 1 is because SSD uses class 0 as a code for "no object" and that needs to be included in the count of classes to track.
Next, change the input\_path and label\_map\_path lines in both the train\_input\_reader and eval\_input\_reader sections to point to our data.  The input\_path should point to /home/ubuntu/tensorflow\_workspace/2020Game/data/2020Game\_train.record-?????-of-00010 for train\_input\_reader and /home/ubuntu/tensorflow\_workspace/2020Game/data/2020Game\_val.record-?????-of-00010 for eval\_input\_reader.  label\_map\_path should be /home/ubuntu/tensorflow\_workspace\2020Game\data\2020Game\_label\_map.pb for both.
Important - the default configs use huge buffers to store input images. This will force the training to run out of memory on most of the systems we have.  Add the following to both the train and eval input reader blocks to reduce mem use

   queue\_capacity: 1000
   min\_after\_dequeue: 1000
   shuffle\_buffer\_size: 256


Another big memory hog is batch\_size, under train\_config.  This needs to be as large as possible without getting out of memory errors (watch for  tensorflow/core/common\_runtime/bfc\_allocator.cc error spam).  There's no real guidance here, somewhere in the mid single digits to maybe 12-15 on the high end?

Next, make sure num\_steps is commented out. This is how many iterations the model is trained for, and the number in the provided configs typically seems too small.

If a fine\_tune\_checkpoint is defined for the model, you might have to download it. See tf1\_detection\_zoo.md for links for those files, and look at the commented section at the top of train.sh for an example of how to use those files.

learning\_rate might need to be updated. The learning rate is how much weights change per step. Typically the configs set up a method to have the LR decay as training progresses. That is, the LR starts at a large value and decreses as training continues, with the idea that big jumps are useful initially to get close to the correct weights and then smaller changes later on allow fine-grained tuning of the weights.

There are two different learning rate decay strategies I've seen so far, each requiring slightly different approaches.

One uses an exponenetial decay.  Here, the initial\_learning\_rate is the rate at the start, and then is reduced by a factor of decay\_factor ^ (current training step / decay\_steps).  For these, a decay factor around 0.9 to 0.95 and a decay\_steps value on the order of 10000 seem to work.  Given we train for ~200K to 300K steps, this gives a good decrease in LR without forcing the LR to basically 0 before training completes.  

Another uses a cosine curve, starting at the learning\_rate\_base value and decaing to 0 after total\_steps.  Key here is that once total\_steps is exceeded, the learning rate is 0 and training stops. So make sure the total steps is bigger than the expected number of training steps needed. In our case, 150,000 to 300,000 seems reasonable, pending further investigation.


