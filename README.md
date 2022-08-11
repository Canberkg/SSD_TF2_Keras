RE-IMPLEMENTATION OF SSD

This repository contains a re-implementation of Single Shot Detector [1] built with Tensorflow 2+/Keras.

Contents

* Dataset Preparation
* Training
* Testing
* To-Do List

Dataset Preparation

The VOC2012 dataset was used in this study, but any dataset with annotations prepared in VOC format can be used. If you also want to use VOC2012 for training and testing, the following steps should be followed carefully:
1. Download training/validation data of VOC2012 Dataset from : http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
2. Unzip the downloaded file and there will be a file named Annotations with XML extension annotations found for each image. All these XML files will be converted to JSON files using xml_to_json.py.
3. Create a new file with the name of your choice for the JSON files and set the directory paths of XML_FILEPATH and JSON_FILEPATH accordingly in the Config.py. Then, run the xml_to_json.py.
4. The VOC2012 file also contains a file called ImageSets. In this file, the names of the training and validation data are contained in two txt files named train and val respectively. Both of these txt files are located in ImageSets\Main. Set TRAIN_SPLIT and VALID_SPLIT in the Config.py accordingly to these path directories. Finally, set the IMG_Filepath to the path of the JPEGImages file inside the VOC 2012 file.
5. After doing all these steps, run the voc_train_valid_split.py. If everything is completed properly, the data should be separated according to the specified files and ready for training.
Training

Before training, all hyperparameters can be set in Config.py. Also, Model name and checkpoint directory name should be set according to your preference. Finally, training can be started by running train.py.

Testing

Before testing, TEST_MODEL_NAME and TEST_IMAGE should be set in the Config.py. Then run the test.py to get the results.

To-Do List


