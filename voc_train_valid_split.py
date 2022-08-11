import os
import shutil
from cv2 import cv2
from Config import cfg_300

def Copy_and_Save (set_path,img_path,annotation_path,new_img_path,new_annotation_path):
    with open(set_path) as f:
        file_names = f.readlines()
    for file in file_names:
        image=cv2.imread(os.path.join(img_path,"{}.jpg".format(file.split('\n')[0])))
        json_file=os.path.join(annotation_path,"{}.json".format(file.split('\n')[0]))
        shutil.copy2(src=json_file, dst=new_annotation_path)
        cv2.imwrite(os.path.join(new_img_path,"{}.jpg".format(file.split('\n')[0])),image)

if __name__ == '__main__':

    IMG_PATH    = cfg_300['IMG_PATH']
    LABEL_PATH  = cfg_300['JSON_ANNOTATION']
    TRAIN_SPLIT = cfg_300['TRAIN_SPLIT']
    VALID_SPLIT = cfg_300['VALID_SPLIT']
    TRAIN_IMG   = cfg_300['TRAIN_IMG']
    VALID_IMG   = cfg_300['VALID_IMG']
    TRAIN_LABEL = cfg_300['TRAIN_LABEL']
    VALID_LABEL = cfg_300['VALID_LABEL']

    set_train_split = TRAIN_SPLIT
    set_val_split   = VALID_SPLIT
    img_path        = IMG_PATH
    label_path      = LABEL_PATH
    train_img       = TRAIN_IMG
    valid_img       = VALID_IMG
    train_label     = TRAIN_LABEL
    valid_label     = VALID_LABEL

    Copy_and_Save(set_path=set_train_split,img_path=img_path,annotation_path=label_path,new_img_path=train_img,new_annotation_path=train_label)
    Copy_and_Save(set_path=set_val_split, img_path=img_path, annotation_path=label_path, new_img_path=valid_img, new_annotation_path=valid_label)