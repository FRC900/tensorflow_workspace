  trained\_retinanet - stock retinanet (ssd\_mobilenet\_v1\_fpn\_shared\_box\_predictor\_640x640\_coco14\_sync.config) with tweaks to nms threshold
  trained\_retinanet\_400 - same as above, input size changed to 415x415 (400x400 threw an error). Probably need to retrain with a larger number of iterations, change optimizer count to match otherwise LR will go to 0
  trained\_retinanet\_mobilenet\_v2\_400 - same as above, but using mobilenetv2 vs. default of v1 as feature extractor. Seems to have a bit better performance
  trained\_ssd\_mobilenet\_v2\_512x512 - Stock mobilenet\_v2\_coco but size changed to 512x512
  trained\_ssd\_mobilenet\_v2\_coco\_focal\_loss - Stock 300x300 sdd mobilenet v2 but using focal loss from FPN
  trained\_ssd\_mobilenet\_v2\_coco\_focal\_loss\_512x512 - Mobilenet ssd v2, 512x512, focal loss, along with cosine LR decay with rmsprop. rmsprop was unintentional, meant to copy full optimizer block from FPN which includes momentum optimizer. Need to retrain a model using cosine LR with momentum, and train for more iterations
