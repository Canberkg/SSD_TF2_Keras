import tensorflow as tf
from Config import NUM_CLASSES,ASPECT_RATIOS_512,SAVED_DIR,BATCH_SIZE
from Utils.encoders import load_mdl



class SSD512(tf.keras.Model):
    def __init__(self):
        super(SSD512, self).__init__()

        self.saved_model_dir=SAVED_DIR
        self.aspect_ratios=ASPECT_RATIOS_512
        self.NUM_CLASSES=NUM_CLASSES

        self.backbone = load_mdl(inputs=tf.keras.Input(shape=(512,512,3),batch_size=BATCH_SIZE),num_class=NUM_CLASSES,model_type='VGG16')
        self.learnable_factor = self.add_weight(shape=(BATCH_SIZE, 1, 1, 512), dtype=tf.float32,initializer=tf.keras.initializers.Ones(), trainable=True)


        self.Conv6_1 = tf.keras.layers.Conv2D(filters=1024,kernel_size=(3,3),strides=(1,1),dilation_rate=(6,6),padding="same",name="Conv6_1")
        self.Conv7_1 = tf.keras.layers.Conv2D(filters=1024,kernel_size=(1,1),strides=(1,1),dilation_rate=(1,1),padding="same",name="Conv7_1")

        self.Conv8_1 = tf.keras.layers.Conv2D(filters=256,kernel_size=(1,1),strides=(1,1),padding="valid",name="Conv8_1")
        self.Conv8_2 = tf.keras.layers.Conv2D(filters=512,kernel_size=(3,3),strides=(2,2),padding="same",name="Conv8_2")

        self.Conv9_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="valid",name="Conv9_1")
        self.Conv9_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same",name="Conv9_2")

        self.Conv10_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="valid",name="Conv10_1")
        self.Conv10_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same",name="Conv10_2")

        self.Conv11_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="valid",name="Conv11_1")
        self.Conv11_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same",name="Conv11_2")

        self.Conv12_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding="valid",name="Conv12_1")
        self.Conv12_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", name="Conv12_2")

        self.classifier_1 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=0))
        self.classifier_2 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=1))
        self.classifier_3 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=2))
        self.classifier_4 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=3))
        self.classifier_5 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=4))
        self.classifier_6 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=5))
        self.classifier_7 = self.classifier(NUM_BOXES=self.get_NUM_BOXES(idx=6))

    def classifier(self,NUM_BOXES):
        filter_size= NUM_BOXES*(self.NUM_CLASSES+4)
        classifier=tf.keras.layers.Conv2D(filters=filter_size,kernel_size=(3,3),strides=(1,1),padding="same")
        return classifier

    def get_NUM_BOXES(self,idx):
        return len(self.aspect_ratios[idx])

    def call(self, inputs, training=None, mask=None):

        lout_list = self.backbone(inputs,training=training)
        classifier_1_inp=tf.math.l2_normalize(x=lout_list[0],axis=-1,epsilon=1e-12)*self.learnable_factor
        classifier_1_out=self.classifier_1(classifier_1_inp)
        loc_cls_1 = tf.reshape(classifier_1_out, shape=(BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        x=tf.nn.relu(self.Conv6_1(lout_list[1]))
        x=tf.nn.relu(self.Conv7_1(x))
        classifier_2_inp=x
        classifier_2_out=self.classifier_2(classifier_2_inp)
        loc_cls_2 = tf.reshape(classifier_2_out, shape=(BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        x=tf.nn.relu(self.Conv8_1(x))
        x=tf.nn.relu(self.Conv8_2(x))
        classifier_3_inp=x
        classifier_3_out=self.classifier_3(classifier_3_inp)
        loc_cls_3 = tf.reshape(classifier_3_out, shape=(BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        x = tf.nn.relu(self.Conv9_1(x))
        x = tf.nn.relu(self.Conv9_2(x))
        classifier_4_inp = x
        classifier_4_out = self.classifier_4(classifier_4_inp)
        loc_cls_4 = tf.reshape(classifier_4_out, shape=(BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        x = tf.nn.relu(self.Conv10_1(x))
        x = tf.nn.relu(self.Conv10_2(x))
        classifier_5_inp = x
        classifier_5_out = self.classifier_5(classifier_5_inp)
        loc_cls_5 = tf.reshape(classifier_5_out, shape=(BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        x = tf.nn.relu(self.Conv11_1(x))
        x = tf.nn.relu(self.Conv11_2(x))
        classifier_6_inp = x
        classifier_6_out = self.classifier_6(classifier_6_inp)
        loc_cls_6 = tf.reshape(classifier_6_out, shape=(BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        x = tf.nn.relu(self.Conv12_1(x))
        x = tf.nn.relu(self.Conv12_2(x))
        classifier_7_inp = x
        classifier_7_out = self.classifier_7(classifier_7_inp)
        loc_cls_7 = tf.reshape(classifier_7_out, shape=(BATCH_SIZE, -1, self.NUM_CLASSES + 4))

        loc_cls = tf.concat([loc_cls_1, loc_cls_2, loc_cls_3, loc_cls_4,loc_cls_5,loc_cls_6,loc_cls_7], axis=1)

        return loc_cls

    def model(self,inputs):

        return tf.keras.Model(inputs=[inputs], outputs=self.call(inputs))
