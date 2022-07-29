import tensorflow as tf
import os

class VGG16(tf.keras.Model):
    def __init__(self,num_classes):
        super(VGG16, self).__init__()

        self.NUM_Classes=num_classes

        self.Conv1_1 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=1,padding="same",name="block1_conv1")
        self.Conv1_2 = tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=1,padding="same",name="block1_conv2")
        self.pool_1=tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding="same",name="block1_pool")

        self.Conv2_1 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=1,padding="same",name="block2_conv1")
        self.Conv2_2 = tf.keras.layers.Conv2D(filters=128,kernel_size=(3,3),strides=1,padding="same",name="block2_conv2")
        self.pool_2=tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding="same",name="block2_pool")

        self.Conv3_1 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=1,padding="same",name="block3_conv1")
        self.Conv3_2 = tf.keras.layers.Conv2D(filters=256,kernel_size=(3,3),strides=1,padding="same",name="block3_conv2")
        self.Conv3_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1, padding="same",name="block3_conv3")
        self.pool_3=tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding="same",name="block3_pool")

        self.Conv4_1 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=1,padding="same",name="block4_conv1")
        self.Conv4_2 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=1,padding="same",name="block4_conv2")
        self.Conv4_3 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=1,padding="same",name="block4_conv3")
        self.pool_4=tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2,padding="same",name="block4_pool")

        self.Conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1,padding="same",name="block5_conv1")
        self.Conv5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1,padding="same",name="block5_conv2")
        self.Conv5_3 = tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1,padding="same",name="block5_conv3")
        self.pool_5=tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=1,padding="same",name="block5_pool")



    def call(self, inputs,training=None, mask=None):



        #First Layer of VGG16
        x=self.Conv1_1(inputs)
        x=tf.nn.relu(x)
        x=self.Conv1_2(x)
        x=tf.nn.relu(x)
        l1_out = x
        x=self.pool_1(x)


        # Second Layer of VGG16
        x = self.Conv2_1(x)
        x = tf.nn.relu(x)
        x = self.Conv2_2(x)
        x = tf.nn.relu(x)
        l2_out = x
        x = self.pool_2(x)


        # Third Layer of VGG16
        x = self.Conv3_1(x)
        x = tf.nn.relu(x)
        x = self.Conv3_2(x)
        x = tf.nn.relu(x)
        x=self.Conv3_3(x)
        x=tf.nn.relu(x)
        l3_out = x
        x = self.pool_3(x)


        # Fourth Layer of VGG16
        x = self.Conv4_1(x)
        x = tf.nn.relu(x)
        x = self.Conv4_2(x)
        x = tf.nn.relu(x)
        x = self.Conv4_3(x)
        x = tf.nn.relu(x)
        l4_out=x
        x = self.pool_4(x)


        #Fifth Layer of VGG16
        x = self.Conv5_1(x)
        x = tf.nn.relu(x)
        x = self.Conv5_2(x)
        x = tf.nn.relu(x)
        x = self.Conv5_3(x)
        x = tf.nn.relu(x)
        l5_out = x
        x = self.pool_5(x)

        return [l4_out,x]

    def model(self,inputs):

        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))