from Primary.DataPrep.class_dict import class_dict
cfg_300 = {
    ## Training Settings
    'BATCH_SIZE' :  1,
    'EPOCH'      :  50,
    'IMG_WIDTH'  :  300,
    'IMG_HEIGHT' :  300,
    'OPTIMIZER'  :  'ADAM',
    'LR'         :  0.001,
    'LR_SCHEDULER' : False,
        'DECAY_STEPS' : 1000,
        'DECAY_RATE'  : 0.96,
    'NUM_CLASS'  : len(class_dict.keys()),

    ## Test Settings
    "TEST_MODEL_NAME" : "Model_Test_3",
    "TEST_IMAGE"      : "D:\\PersonalResearch\\Projects\\Datasets\\VOC2012\\Img\\val\\2008_000059.jpg",

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
    'SAVE_DIR'   : "D:\\PersonalResearch\\Projects\\Models\\SSD",
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
