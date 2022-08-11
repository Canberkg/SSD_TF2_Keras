# Single-Shot Detector (SSD)

This repository contains the re-implementation of SSD [1] built with Tensorflow 2+/Keras.

## Contents

* Data Preparation
* Training
* Testing
* To-do List
* Notes

## Data Preparation

VOC 2012 dataset is used in this repo, but any dataset with annotations prepared in voc format can be easily integrated. The following steps should be followed carefully to prepare the dataset:
1. Download the VOC2012 training/validation dataset from : http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
2. Unzip the file. There are three important directory that we are going to use. These are Annotations, JPEGImages and ImageSets.
3. Annotations contain XML extension annotation files for each image. This repo supports JSON files. Therefore, create a directory with the name of your choice in the VOC2012 directory for the JSON extension annotation files. Then copy and paste the directory path of the Annotations file next to the XML ANNOTATION in Config.py in the repo. In the same way, copy and paste the path of the newly created JSON directory next to the JSON_ANNOTATION.
4. The JPEGImages file contains all images for both training and validation. Copy and paste the directory path of the JPEGImages file next to the IMG_PATH in Config.py
5. Finally, ImageSets\Main contains two txt files train and val. These files contain the names of the images of the training and validation images, respectively. Copy and paste the directory path of each txt file next to TRAIN_SPLIT and VALID_SPLIT in Config.py respectively
6. After completing all these steps, first run xml_to_json.py to generate the JSON versions of the annotations. Then run voc_train_valid_split.py to separate each image for training and validation.

## Training

Before training, the MODEL_NAME and CKPT_DIR fields in Config.py should be set.In addition, other hyperparameters such as learning rate and batch size can also be adjusted from within the same file.After that, training can be started simply by running train.py.

## Testing

For testing, the name of the model used for testing and the path to the image must be set in Config.py.. Then run test.py to get the results.

## To-Do List

- [ ] Command-Line Interface
- [ ] Data Augmentation

##  Notes

* Since I only have google colab pro to train and test my network, it takes a lot of time for me to train and see the results of the training. I have observed that the implementation has an overfitting issue that can be overcome by integrating data augmentation into the data generation process. However, it may take time for me to fix the problem due to the situation I mentioned. I will also share the weights of the model when I succeed in obtaining a model that performs well in both training and validation data. If you have any feedback please let me know, I would love to discuss about it and improve the repository with your feedback.

## Reference
[1]LIU, Wei, et al. Ssd: Single shot multibox detector. In: European conference on computer vision. Springer, Cham, 2016. p. 21-37.

https://github.com/bubbliiiing/ssd-tf2
