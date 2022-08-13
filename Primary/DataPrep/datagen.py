import os
import json
import random
import tensorflow as tf
import numpy as np
import cv2
from Primary.DataPrep.VOC import VOC

class data_gen(tf.keras.utils.Sequence):
    """Data Generator Class

    Attributes :
        Img_Path      : Path of Image Directory
        Label_Path    : Path of Label Directory
        Img_Width     : Width of input accepted by the network (Int)
        Img_Height    : Height of input accepted by the network (Int)
        Batch_Size    : Length of a  Batch
        Num_Classes   : Number of Classes
        Shuffle       : Whether Shuffle the data or not (boolean)
        Augmentation  : Augmentation Techniques (Horizontal/Vertical Flip : "FLIP" )

    """
    def __init__(self, Img_Path, Label_Path, Img_Width, Img_Height, Batch_Size, Num_Classes, Shuffle=False,Augmentation=None):
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
        self.Augmentation=Augmentation
        self.shuffle = Shuffle

    def num_images(self):
        """Find the total number of image
        Params:
        Return:
           Total number of image (int)
        """
        return len(self.Img_list) - (len(self.Img_list) % self.Batch_Size)

    def on_epoch_end(self):
        """Apply after completion of each epoch
        Params:
        Return:
            Shuffled indices if Shuufle is True (int)
        """
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __len__(self):
        """Find the original width of image
        Params:
        Return:
            Total number of batches (int)
        """
        return (len(self.Img_list) // self.Batch_Size)

    def __getitem__(self, index):
        """Get batch of images and labels
        Params:
            index: Index of a batch
        Return:
            x: Batch of images (Tuple)
            y: Batch of labels (List)
        """
        index = self.index[index * self.Batch_Size:(index + 1) * self.Batch_Size]
        batch = [self.indices[i] for i in index]

        x, y = self.__getdata__(batch)

        return x, y

    def augmentation_flip(self,img,labels):
        """Perform horizontal/vertical flip operations with a random probability
        Params:
            img: Image array (Array)
            labels: Label Tensor (Tensor)
        Return:
            img: Augmented Image
            labels: Augmented Labels
        """
        augmentation_prob=random.random()
        if augmentation_prob > 0.2 and augmentation_prob <= 0.6:

            img = img[::-1,:,:]
            labels = tf.stack([labels[:,0],1-labels[:,1],labels[:,2],labels[:,3],labels[:,4]],axis=1)
        elif augmentation_prob > 0.6 and augmentation_prob <= 1:

            img = img[:, ::-1, :]
            labels = tf.stack([1-labels[:, 0], labels[:, 1], labels[:, 2], labels[:, 3], labels[:, 4]], axis=1)
        return img,labels

    def __getdata__(self, batch):
        """Generate batch of images and labels
        Params:
            batch: Indices of Images
        Return:
            Batch of image (Tuple),Batch of labels (List)
        """
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
                img_normalized = tf.keras.applications.vgg16.preprocess_input(Img_arr)

                label_ind = self.Label_list.index(label)
                f = open(os.path.join(self.Label_Path, self.Label_list[label_ind]))
                data = json.load(f)
                voc = VOC(json_data=data,IMG_WIDTH=self.Img_Width,IMG_HEIGHT=self.Img_Height)
                gt_boxes = voc.obj_to_gt()

                if self.Augmentation == "FLIP":
                    img_normalized,gt_boxes=self.augmentation_flip(img=img_normalized,labels=gt_boxes)

                Images.append(img_normalized)
                Labels[k] = gt_boxes

            else:
                raise Exception('Image and Annotation not the same!')

        return tuple((tf.stack(Images, axis=0), Labels))