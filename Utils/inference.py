import tensorflow as tf
from Utils.utils import nms
from Primary.BoundingBox.DefaultBoxes import DefaultBoxes

tf.random.set_seed(1234)

class inference(object):
    """Inference Class

    Attributes :
            model         : Model
            NUM_CLASSES   : Number of Classes  (Int)
            IMG_HEIGHT    : Height of input accepted by the network (Int)
            IMG_WIDTH     : Width of input accepted by the network (Int)
            FEATURE_MAPS  : List of feature map shapes  (List)
            ASPECT_RATIOS : List of aspect ratios for anchor boxes (List)
            SIZES         : List of scales for anchor boxes (List)
    """
    def __init__(self,model,NUM_CLASSES,IMG_HEIGHT,IMG_WIDTH,FEATURE_MAPS,ASPECT_RATIOS,SIZES):
        self.model=model
        self.IMG_HEIGHT=IMG_HEIGHT
        self.IMG_WIDTH=IMG_WIDTH
        self.NUM_CLASSES=NUM_CLASSES
        self.FEATURE_MAPS=FEATURE_MAPS
        self.ASPECT_RATIOS=ASPECT_RATIOS
        self.SIZES=SIZES

    def ssd_prediction(self,image):
        """Prediction by using specified model
        Params:
            image: Image array (Array)
        Return:
            ssd_pred: predicted offset bounding boxes (Tensor)
        """
        ssd_pred=self.model(image)
        return ssd_pred

    def filter_out_background_boxes(self,pred_box):
        """Filter out background boxes
        Params:
            pred_box: predicted bounding boxes (Tensor)
        Return:
            object_exist: Whether object exist or not (boolean)
            non_background_boxes: If object exist, bounding boxes for wanted objects (Tensor)
        """
        pred_box_classes=pred_box[...,:self.NUM_CLASSES]
        sftmx=tf.nn.softmax(pred_box_classes)
        true_classes=tf.argmax(sftmx,axis=-1)
        non_background_indices=tf.where(true_classes!=0)
        object_exist=tf.equal(non_background_indices.shape[0],0)
        if object_exist == False:
            non_background_boxes=tf.gather(pred_box,indices=non_background_indices)
            return object_exist,non_background_boxes
        else:
            return object_exist,pred_box

    def decode_offsets_to_true_boxes(self,offset_pred):
        """Decode founded offset bounding boxes
        Params:
            offset_pred: predicted offset bounding boxes (Tensor)
        Return:
            true_predictions: Decoded bounding boxes (Tensor)
        """
        pred_classes = tf.reshape(tensor=offset_pred[..., :self.NUM_CLASSES], shape=(-1, self.NUM_CLASSES))
        pred_boxes= tf.reshape(tensor=offset_pred[..., self.NUM_CLASSES:], shape=(-1, 4))
        default_boxes=DefaultBoxes(Feature_Maps=self.FEATURE_MAPS,IMG_WIDTH=self.IMG_WIDTH,IMG_HEIGHT=self.IMG_HEIGHT,ASPECT_RATIOS=self.ASPECT_RATIOS,SIZES=self.SIZES).generate_default_boxes()


        true_x_c=(pred_boxes[...,0]*default_boxes[...,2]*0.1)+default_boxes[...,0]
        true_y_c = (pred_boxes[..., 1] * default_boxes[..., 3]*0.1) + default_boxes[..., 1]
        true_w = tf.math.exp(pred_boxes[...,2]*0.2) *default_boxes[..., 2]
        true_h = tf.math.exp(pred_boxes[...,3]*0.2) *default_boxes[..., 3]

        true_x_c = tf.expand_dims(true_x_c,axis=-1)
        true_y_c = tf.expand_dims(true_y_c, axis=-1)
        true_w = tf.expand_dims(true_w, axis=-1)
        true_h = tf.expand_dims(true_h, axis=-1)

        true_boxes_cord=tf.concat(values=[true_x_c,true_y_c,true_w,true_h],axis=-1)
        true_predictions=tf.concat(([pred_classes,true_boxes_cord]),axis=-1)

        return true_predictions


    def detected_boxes(self,image):
        """Detect existing objects
        Params:
            image: Image (Tensor)
        Return:
            object_exist: Whether object exist or not (boolean)
            boxes: Remaining boxes after the NMS (Tensor)
            scores: Scores of the remaining boxes after the NMS (Tensor)
            classes: Classes of the remaining boxes after the NMS (Tensor)
        """
        pred_boxes=self.ssd_prediction(image=image)
        true_boxes=self.decode_offsets_to_true_boxes(offset_pred=pred_boxes)
        object_exist,non_background_boxes=self.filter_out_background_boxes(pred_box=true_boxes)
        if object_exist==False:
            class_scores=non_background_boxes[...,:self.NUM_CLASSES]
            true_cords=non_background_boxes[...,self.NUM_CLASSES:]

            #resize
            resized_c_x_min = (true_cords[...,0]- (0.5*true_cords[..., 2]))
            resized_c_y_min = (true_cords[..., 1]- (0.5*true_cords[..., 3]))
            resized_c_x_max= (true_cords[...,0]+ (0.5*true_cords[..., 2]))
            resized_c_y_max = (true_cords[...,1]+ (0.5*true_cords[..., 3]))

            resized_true_boxes = tf.stack(values=[resized_c_y_min[...,0], resized_c_x_min[...,0], resized_c_y_max[...,0], resized_c_x_max[...,0]], axis=-1)

            boxes,scores,classes=nms(BBoxes=resized_true_boxes,BBox_scores=class_scores[:,0,:])

            return object_exist, boxes,scores,classes
        else:
            return object_exist, tf.zeros(shape=(1, 4)), tf.zeros(shape=(1,)), tf.zeros(shape=(1,))








