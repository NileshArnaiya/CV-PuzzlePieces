# Tino Puzzle Piece Image Segmentation

### Dataset Creation

The dataset was created by Richard klein, Devon Jarvis, Nathan Michlo. 

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



