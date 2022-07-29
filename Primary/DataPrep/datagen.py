import os
import json
import tensorflow as tf
import numpy as np
import cv2

from Primary.DataPrep.VOC import VOC

class data_gen(tf.keras.utils.Sequence):
    def __init__(self, Img_Path, Label_Path, Img_Width, Img_Height, Batch_Size, Num_Classes, Shuffle=False):
        self.Img_Path = Img_Path
        self.Label_Path = Label_Path
        self.Img_list = os.listdir(self.Img_Path)
        self.Label_list = os.listdir(self.Label_Path)
        self.Img_Width = Img_Width
        self.Img_Height = Img_Height
        self.Batch_Size = Batch_Size
        self.Num_Classes = Num_Classes
        self.indices = range(0, len(self.Img_list) - (len(self.Img_list) % self.Batch_Size))
        self.index = np.arange(0, len(self.Img_list) - (len(self.Img_list) % self.Batch_Size))
        self.shuffle = Shuffle

    def num_images(self):
        return len(self.Img_list) - (len(self.Img_list) % self.Batch_Size)

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        return (len(self.Img_list) // self.Batch_Size)

    def __getitem__(self, index):
        index = self.index[index * self.Batch_Size:(index + 1) * self.Batch_Size]
        batch = [self.indices[i] for i in index]

        x, y = self.__getdata__(batch)

        return x, y

    def __getdata__(self, batch):

        Images = []
        Labels = {}

        batch_of_images = [self.Img_list[i] for i in batch]

        for k in range(len(batch)):
            label_splt = batch_of_images[k].split('.')[0]
            label="{}.json".format(label_splt)
            if label in self.Label_list:

                Img = cv2.imread(filename=os.path.join(self.Img_Path, batch_of_images[k]))
                Img = cv2.resize(Img, (self.Img_Width, self.Img_Height), interpolation=cv2.INTER_NEAREST)
                Img_arr = np.asarray(Img, dtype=np.float32)
                Img_normalized = tf.keras.applications.vgg16.preprocess_input(Img_arr)
                Images.append(Img_normalized)

                label_ind = self.Label_list.index(label)
                f = open(os.path.join(self.Label_Path, self.Label_list[label_ind]))
                data = json.load(f)
                voc = VOC(json_data=data,IMG_WIDTH=self.Img_Width,IMG_HEIGHT=self.Img_Height)
                gt_boxes = voc.obj_to_gt()
                Labels[k] = gt_boxes

            else:
                raise Exception('Image and Annotation not the same!')

        return tuple((tf.stack(Images, axis=0), Labels))