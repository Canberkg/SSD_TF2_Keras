import numpy as np
import tensorflow as tf
from Utils.encoders import load_mdl


class Normalize(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        self.axis = 3
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [tf.keras.layers.InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = tf.keras.backend.variable(init_gamma, name='{}_gamma'.format(self.name))

    def call(self, x, mask=None):
        output = tf.keras.backend.l2_normalize(x, self.axis)
        output *= self.gamma
        return output

class SSD300(tf.keras.Model):
    def __init__(self,ASPECT_RATIOS,BATCH_SIZE,NUM_CLASS):
        super(SSD300, self).__init__()

        self.aspect_ratios=ASPECT_RATIOS
        self.NUM_CLASSES=NUM_CLASS
        self.BATCH_SIZE=BATCH_SIZE

        self.backbone = load_mdl(inputs=tf.keras.Input(shape=(300,300,3),batch_size=BATCH_SIZE),num_class=self.NUM_CLASSES,model_type='VGG16')
        self.l2_normalize = Normalize(scale=20,name='conv4_3_norm')

        self.Conv6_1 = tf.keras.layers.Conv2D(filters=1024,kernel_size=(3,3),strides=(1,1),dilation_rate=(6,6),padding="same",name="Conv6_1")
        self.Conv7_1 = tf.keras.layers.Conv2D(filters=1024,kernel_size=(1,1),strides=(1,1),dilation_rate=(1,1),padding="same",name="Conv7_1")

        self.Conv8_1 = tf.keras.layers.Conv2D(filters=256,kernel_size=(1,1),strides=(1,1),padding="valid",name="Conv8_1")
        self.Conv8_2 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(2,2),padding="same",name="Conv8_2")

        self.Conv9_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="valid",name="Conv9_1")
        self.Conv9_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same",name="Conv9_2")

        self.Conv10_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="valid",name="Conv10_1")
        self.Conv10_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="valid",name="Conv10_2")

        self.Conv11_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="valid",name="Conv11_1")
        self.Conv11_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="valid",name="Conv11_2")

        self.classifier_1=self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=0),name='classifier_1')
        self.classifier_2 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=1),name='classifier_2')
        self.classifier_3 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=2),name='classifier_3')
        self.classifier_4 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=3),name='classifier_4')
        self.classifier_5 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=4),name='classifier_5')
        self.classifier_6 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=5),name='classifier_6')

    def classifier(self,NUM_BOXES,name):
        filter_size= NUM_BOXES*(self.NUM_CLASSES+4)
        classifier=tf.keras.layers.Conv2D(filters=filter_size,kernel_size=(3,3),strides=(1,1),padding="same",name=name)
        return classifier

    def get_NUM_BOXES(self,idx):
        return len(self.aspect_ratios[idx])+1

    def call(self, inputs, training=None, mask=None):

        lout_list = self.backbone(inputs, training=training)

        classifier_1_inp=self.l2_normalize(lout_list[0])
        classifier_1_out=self.classifier_1(classifier_1_inp)
        loc_cls_1 = tf.reshape(classifier_1_out, shape=(self.BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        x=tf.nn.relu(self.Conv6_1(lout_list[1]))
        x=tf.nn.relu(self.Conv7_1(x))
        classifier_2_inp=x
        classifier_2_out=self.classifier_2(classifier_2_inp)
        loc_cls_2 = tf.reshape(classifier_2_out, shape=(self.BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        x=tf.nn.relu(self.Conv8_1(x))
        x=tf.nn.relu(self.Conv8_2(x))
        classifier_3_inp=x
        classifier_3_out=self.classifier_3(classifier_3_inp)
        loc_cls_3 = tf.reshape(classifier_3_out, shape=(self.BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        x = tf.nn.relu(self.Conv9_1(x))
        x = tf.nn.relu(self.Conv9_2(x))
        classifier_4_inp = x
        classifier_4_out = self.classifier_4(classifier_4_inp)
        loc_cls_4 = tf.reshape(classifier_4_out, shape=(self.BATCH_SIZE, -1, self.NUM_CLASSES + 4))


        x = tf.nn.relu(self.Conv10_1(x))
        x = tf.nn.relu(self.Conv10_2(x))
        classifier_5_inp = x
        classifier_5_out = self.classifier_5(classifier_5_inp)
        loc_cls_5 = tf.reshape(classifier_5_out, shape=(self.BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        x = tf.nn.relu(self.Conv11_1(x))
        x = tf.nn.relu(self.Conv11_2(x))
        classifier_6_inp = x
        classifier_6_out = self.classifier_6(classifier_6_inp)
        loc_cls_6 = tf.reshape(classifier_6_out, shape=(self.BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        loc_cls = tf.concat([loc_cls_1, loc_cls_2, loc_cls_3, loc_cls_4, loc_cls_5, loc_cls_6], axis=1)

        return loc_cls

    def model(self, inputs):
        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))

