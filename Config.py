from Primary.DataPrep.class_dict import class_dict
cfg_300 = {
    ## Training Settings
    'BATCH_SIZE' :  16,
    'EPOCH'      :  50,
    'IMG_WIDTH'  :  300,
    'IMG_HEIGHT' :  300,
    'OPTIMIZER'  :  'ADAM', # Can be set as ADAM or SGD
    'LR'         :  0.001,  # if the LR_SCHEDULER set as true, LR will be init point for the scheduler otherwise will be set as a constant learning rate
    'LR_SCHEDULER' : False, #If it is specified as False, dont have to change the Decay steps or rate!
        'DECAY_STEPS' : 1000,
        'DECAY_RATE'  : 0.96,
    'NUM_CLASS'  : len(class_dict.keys()),

    ## Test Settings
    "TEST_MODEL_NAME" : "The name of the model to be used for testing",
    "TEST_IMAGE"      : "Path of the image to be used for testing",

    ## Anchor Box Settings
    'ASPECT_RATIOS' : [[1.0, 2.0, 0.5],
                      [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                      [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                      [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                      [1.0, 2.0, 0.5],
                      [1.0, 2.0, 0.5]],
    'SIZES'         : [(30,60),(60,111),(111,162),(162,213),(213,264),(264,315)],
    'IOU_THRESHOLD' : 0.5,
    'FEATURE_MAPS'  : [(38,38),(19,19),(10,10),(5,5),(3,3),(1,1)],

    ## Model Save and Checkpoint Settings
    'SAVE_DIR'   : "Saved_Model",
    'MODEL_NAME' : 'Name of the Model',
    'CKPT_DIR'   : 'Name of the Checkpoint Directory',

    ## Dataset Settings
    'TRAIN_IMG'   : "Dataset/images/train",
    'VALID_IMG'   : "Dataset/images/valid",
    "TRAIN_LABEL" : "Dataset/annotations_json/train",
    "VALID_LABEL" : "Dataset/annotations_json/valid",
    'SHUFFLE' : True,

    ## Dataset Prepearation
    'IMG_PATH'       : "Directory Path of Images",
    'XML_ANNOTATION' : "Directory Path of XML Annotations",
    'JSON_ANNOTATION': "Directory Path of JSON Annotations",
    'TRAIN_SPLIT'    : "Directory Path of training set txt file which contains the name of the all the image names belong to training",
    'VALID_SPLIT'    : "Directory Path of training set txt file which contains the name of the all the image names belong to validation",

}
