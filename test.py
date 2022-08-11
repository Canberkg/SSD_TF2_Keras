import os
import tensorflow as tf
import numpy as np
from cv2 import cv2

from Config import cfg_300
from Detection.SSD300 import SSD300
from Utils.inference import inference
from Utils.utils import Visualize_BB
np.random.seed(1234)
tf.random.set_seed(1234)

if __name__ == "__main__":

    BATCH_SIZE = cfg_300['BATCH_SIZE']
    ASPECT_RATIOS = cfg_300['ASPECT_RATIOS']
    IMG_WIDTH       = cfg_300['IMG_WIDTH']
    IMG_HEIGHT      = cfg_300['IMG_HEIGHT']
    NUM_CLASSES     = cfg_300['NUM_CLASS']
    FEATURE_MAPS    = cfg_300['FEATURE_MAPS']
    SIZES           = cfg_300['SIZES']
    SAVE_DIR        = cfg_300['SAVE_DIR']
    TEST_MODEL_NAME = cfg_300['TEST_MODEL_NAME']
    TEST_IMG_PATH   = cfg_300["TEST_IMAGE"]

    ssd_model = SSD300(ASPECT_RATIOS=ASPECT_RATIOS,BATCH_SIZE=BATCH_SIZE,NUM_CLASS=NUM_CLASSES)
    ssd_model.build(input_shape=(1,300,300,3))
    ssd_model.load_weights(os.path.join(SAVE_DIR,"{}.h5".format(TEST_MODEL_NAME)))
    ssd_model.trainable=False
    ssd_model.summary()

    Images=[]
    croped_Images = []
    Imaged = cv2.imread(filename=TEST_IMG_PATH)
    Img = cv2.resize(Imaged, (300,300), interpolation=cv2.INTER_NEAREST)
    Img_arr = np.asarray(Img, dtype=np.float32)


    Img_normalized = tf.keras.applications.vgg16.preprocess_input(Img_arr)
    Images.append(Img_normalized)
    stacked_image=tf.stack(Images,axis=0)
    inference_process = inference(model=ssd_model,NUM_CLASSES=NUM_CLASSES,IMG_HEIGHT=IMG_HEIGHT,IMG_WIDTH=IMG_WIDTH,FEATURE_MAPS=FEATURE_MAPS,ASPECT_RATIOS=ASPECT_RATIOS,SIZES=SIZES)
    is_object_exist, boxes, scores,classes = inference_process.detected_boxes(image=stacked_image)

    if is_object_exist==False:
        image_norm = cv2.imread(filename=TEST_IMG_PATH)
        image=cv2.resize(image_norm,dsize=(300,300))

        b_x_c = (boxes[..., 1]+((boxes[..., 3]-boxes[..., 1])/2))
        b_y_c = (boxes[..., 0] +((boxes[..., 2]-boxes[..., 0])/2))
        b_w = ((boxes[..., 3]-boxes[..., 1]))
        b_h = ((boxes[..., 2]-boxes[..., 0]))

        b_x_c = tf.expand_dims(b_x_c, axis=-1)
        b_y_c = tf.expand_dims(b_y_c, axis=-1)
        b_w = tf.expand_dims(b_w, axis=-1)
        b_h = tf.expand_dims(b_h, axis=-1)

        sc=tf.expand_dims(scores,axis=-1)
        box=tf.concat(values=[b_x_c,b_y_c,b_w,b_h,sc],axis=-1)
        sorted_boxes=tf.gather(box,tf.math.top_k(box[...,-1],k=1,sorted=True).indices)
        np_sorted = box[..., :4].numpy()
        np_sorted[..., 0] = np_sorted[..., 0] * 300
        np_sorted[..., 2] = np_sorted[..., 2] * 300
        np_sorted[..., 1] = np_sorted[..., 1] * 300
        np_sorted[..., 3] = np_sorted[..., 3] * 300
        Visualize_BB(image=image,BB=np_sorted,scores=sc,color=(0,255,0))
        cv2.imshow("image", image)
        cv2.waitKey()
    else:
        print("No objects were detected.")
        image = cv2.imread(filename=TEST_IMG_PATH)
        cv2.imshow("image", image)
        cv2.waitKey()


