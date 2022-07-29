import os
import shutil
from cv2 import cv2

def Copy_and_Save (set_path,img_path,annotation_path,new_img_path,new_annotation_path):
    with open(set_path) as f:
        file_names = f.readlines()
    for file in file_names:
        image=cv2.imread(os.path.join(img_path,"{}.jpg".format(file.split('\n')[0])))
        json_file=os.path.join(annotation_path,"{}.json".format(file.split('\n')[0]))
        shutil.copy2(src=json_file, dst=new_annotation_path)
        cv2.imwrite(os.path.join(new_img_path,"{}.jpg".format(file.split('\n')[0])),image)

if __name__ == '__main__':

    set_train_path="D:\\PersonalResearch\\Projects\\Datasets\\VOC2012\\ImageSets\\Main\\train.txt"
    set_val_path = "D:\\PersonalResearch\\Projects\\Datasets\\VOC2012\\ImageSets\\Main\\val.txt"
    img_path="D:\\PersonalResearch\\Projects\\Datasets\\VOC2012\\JPEGImages"
    label_path = "D:\\PersonalResearch\\Projects\\Datasets\\VOC2012\\Annotations_json"
    train_img_split="D:\\PersonalResearch\\Projects\\Datasets\\VOC2012\\Img\\train"
    valid_img_split = "D:\\PersonalResearch\\Projects\\Datasets\\VOC2012\\Img\\val"
    train_label_split="D:\\PersonalResearch\\Projects\\Datasets\\VOC2012\\Labels\\train"
    valid_label_split = "D:\\PersonalResearch\\Projects\\Datasets\\VOC2012\\Labels\\val"

    #Copy_and_Save(set_path=set_train_path,img_path=img_path,annotation_path=label_path,new_img_path=train_img_split,new_annotation_path=train_label_split)
    Copy_and_Save(set_path=set_val_path, img_path=img_path, annotation_path=label_path, new_img_path=valid_img_split, new_annotation_path=valid_label_split)