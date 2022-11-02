# Jigsaw Puzzle Piece Image Segmentation & Placement Prediction

## Motivation

Despite having only rudimentary exposure to image classification and no exposure to semantic/instance segmentation, I found myself gravitating towards instance segmentation.  Inspired by [this writeup](https://engineering.matterport.com/splash-of-color-instance-segmentation-with-mask-r-cnn-and-tensorflow-7c761e238b46) about using Matterport's Mask R-CNN for pixel-level balloon identification, I started mulling ideas with my good friend and founder of the [codebug bootcamp](https://www.codebug.us/).  A couple of  rabbit holes later, we stumbled upon a couple of puzzles under the coffee table and came up with an initial business question:

***Can you take a photo of a puzzle piece and the photo of its box and predict where in the puzzle it belongs?***

As an avid puzzler growing up, I thought this would be a fun challenge that had several checkpoints (and stretch goals) that allowed me to gauge the feasibility of the task along the way and adjust as needed.

## Project Organization

### Dataset Creation

The dataset was created by taking pieces from 5 puzzles and photographing them in *expected* situations (in the puzzle's box, in one's hand, on a table, etc.).  Given the business application of the desired solution, it did not make sense to photograph these pieces in random situations.  For the training and validation sets, the neural network requires the object outlines to be annotated and classified, so I used the [VGG Image Annotator (VIA)](http://www.robots.ox.ac.uk/~vgg/software/via/) to create these in JSON.

###### Training data

- 48x2 puzzles (RGB + MASKS)
- 34 Training, 7, validation, 7 test 
- Image augmentation 

#### Part I: Baseline 
#### Part 2: GMM
#### Part 3: U-Net

2. ***Semantic Segmentation***: Identifying all the pixels of puzzle piece(s) in the image.

The first part of the project focused on image segmentation and being able to accurately classify and locate a puzzle piece in an image.  To do this, We performed various methods and compared their ROC AUC Score and IOU Scores to see which works best. 
GMM AND U-net clearly outperformed. 

## Resources

#### Instance Segmentation & Other readings we did

- [vidhya - Step-by-Step Introduction to Image Segmentation Techniques](https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/)

- [jeremyjordan - An overview of semantic image segmentation](https://www.jeremyjordan.me/semantic-segmentation/)
- [github - Image Segmentation with tf.keras](https://github.com/tensorflow/models/blob/master/samples/outreach/blogs/segmentation_blogpost/image_segmentation.ipynb)
- [colab - maskrcnn_custom_tf_colab.ipynb]



