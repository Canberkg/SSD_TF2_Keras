from Primary.DataPrep.class_dict import class_dict
cfg_300 = {
    ## Training Settings
    'BATCH_SIZE' :  4,
    'EPOCH'      :  5,
    'IMG_WIDTH'  :  300,
    'IMG_HEIGHT' :  300,
    'OPTIMIZER'  :  'ADAM', # Can be set as ADAM or SGD
    'LR'         :  0.001,  # if the LR_SCHEDULER set as true, LR will be init point for the scheduler otherwise will be set as a constant learning rate
    'LR_SCHEDULER' : False, #If it is specified as False, dont have to change the Decay steps or rate!
        'DECAY_STEPS' : 1000,
        'DECAY_RATE'  : 0.96,
    'NUM_CLASS'  : len(class_dict.keys()),

    ## Anchor Box Settings
    'ASPECT_RATIOS'  : [[1.0, 2.0, 0.5],
                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                       [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                       [1.0, 2.0, 0.5],
                       [1.0, 2.0, 0.5]],
    'MIN_SCALE'     : 0.2,
    'MAX_SCALE'     : 0.9,
    'IOU_THRESHOLD' : 0.5,
    'FEATURE_MAPS'  : [(38,38),(19,19),(10,10),(5,5),(3,3),(1,1)],

    ## Model Save and Checkpoint Settings
    'SAVE_DIR'   : "D:\\PersonalResearch\\Projects\\SSD\\Saved_Model",
    'MODEL_NAME' : 'Model_Test',
    'CKPT_DIR'   : 'ckpt_test',

    ## Dataset Settings
    'TRAIN_IMG'   : "D:\\PersonalResearch\\Projects\\SSD\\Dataset\\images\\train",
    'VALID_IMG'   : "D:\\PersonalResearch\\Projects\\SSD\\Dataset\\images\\valid",
    "TRAIN_LABEL" : "D:\\PersonalResearch\\Projects\SSD\\Dataset\\annotations_json\\train",
    "VALID_LABEL" : "D:\\PersonalResearch\\Projects\SSD\\Dataset\\annotations_json\\valid",
    'SHUFFLE' : True

}
