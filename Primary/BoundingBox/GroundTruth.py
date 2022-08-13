
import tensorflow as tf
from Primary.BoundingBox.DefaultBoxes import DefaultBoxes
from Utils.utils import calculate_iou

class GroundTruth(object):
    """Ground Truth Class

    Attributes :
            BOXES         : List of Ground Truth Boxes  (List)
            FEATURE_MAPS  : List of feature map shapes  (List)
            IMG_WIDTH     : Width of input accepted by the network (Int)
            IMG_HEIGHT    : Height of input accepted by the network (Int)
            IOU_THRESHOLD : IOU threshold for matching process (float)
            ASPECT_RATIOS : List of aspect ratios for anchor boxes (List)
            SIZES         : List of scales for anchor boxes (List)
    """
    def __init__(self,BOXES,FEATURE_MAPS,IMG_WIDTH,IMG_HEIGHT,IOU_THRESHOLD,ASPECT_RATIOS,SIZES):
        self.B_Boxes=BOXES
        self.default_boxes=DefaultBoxes(Feature_Maps=FEATURE_MAPS,IMG_WIDTH=IMG_WIDTH,IMG_HEIGHT=IMG_HEIGHT,ASPECT_RATIOS=ASPECT_RATIOS,SIZES=SIZES)
        self.predicted_boxes=self.default_boxes.generate_default_boxes()
        self.IOU_THRESHOLD=IOU_THRESHOLD

    def positive_negative_iou_metric(self,GT_boxes,Anchor_Boxes):
        """Label all positive negative boxes based on specified IOU Threshold

        Params:
            GT_boxes: Ground Truth Boxes on a single Image (Tensor)
            Anchor_Boxes: Generated Anchor boxes (Tensor)
        Return:
            anchor_state: Contains the relationship between each anchor box and the ground truth box (Tensor)
            max_iou_indexes: Contains the index of ground truth boxes where each anchor box gets the maximum IOU (Tensor)

        """
        iou_scores = calculate_iou(GT_boxes, Anchor_Boxes)

        max_iou_values = tf.math.reduce_max(iou_scores, axis=0)
        max_iou_indexes = tf.math.argmax(iou_scores, axis=0)
        max_iou_indexes = tf.cast(max_iou_indexes, dtype=tf.dtypes.int32)
        max_iou_indexes = tf.expand_dims(max_iou_indexes, axis=-1)

        positive_negative_boxes = tf.where(max_iou_values>= self.IOU_THRESHOLD, 1, -1)
        anchor_state=tf.where(max_iou_values<0.4, 0, positive_negative_boxes)

        anchor_state = tf.cast(anchor_state, dtype=tf.dtypes.int32)

        return anchor_state,max_iou_indexes

    def get_offset(self,gt_box,pred_box):
        """Find offset dimensions between anchor boxes and their corresponding ground truths

        Params:
            gt_box: Ground Truth Boxes on a single Image (Tensor)
            pred_box: Generated Anchor boxes (Tensor)
        Return:
            Every offset dimension between anchor boxes and their corresponding ground truths (Tensor)
        """
        epsilon=0.000000000000001
        gt_x_c,gt_y_c,gt_w,gt_h=gt_box[:,0],gt_box[:,1],gt_box[:,2],gt_box[:,3]
        pred_x_c,pred_y_c,pred_w,pred_h=pred_box[:,0],pred_box[:,1],pred_box[:,2],pred_box[:,3]

        ofst_x = (gt_x_c-pred_x_c)/pred_w
        ofst_y = (gt_y_c - pred_y_c) / pred_h

        ofst_w = tf.math.log(gt_w / (pred_w+epsilon))
        ofst_h = tf.math.log(gt_h/ (pred_h+epsilon))

        return tf.stack([ofst_x,ofst_y,ofst_w,ofst_h],axis=1)/[0.1, 0.1, 0.2, 0.2]
    def get_positive_negative_boxes(self,GT_boxes,Pred_Boxes):
        """Find positive and negative offset boxes

        Params:
            GT_boxes: Ground Truth Boxes on a single Image (Tensor)
            Pred_Boxes: Generated Anchor boxes (Tensor)
        Return:
            positive_negative_box_array: Positive and negative offset boxes for loss calculation (Tensor)
            anchor_state: Contains the relationship between each anchor box and the ground truth box (Tensor)
        """
        GT_BBoxes=GT_boxes[...,:4]
        GT_class=GT_boxes[...,-1]
        GT_BBoxes = tf.cast(GT_BBoxes, dtype=tf.dtypes.float32)
        GT_class = tf.cast(GT_class, dtype=tf.dtypes.int32)


        anchor_state,max_iou_indexes=self.positive_negative_iou_metric(GT_boxes=GT_BBoxes,Anchor_Boxes=Pred_Boxes)
        GT_class=tf.expand_dims(GT_class,axis=-1)

        box_class_index_array=tf.gather(GT_class,indices=max_iou_indexes[:,0])
        box_offset_array=self.get_offset(gt_box=tf.gather(GT_BBoxes[:,...],max_iou_indexes[:,0]),pred_box=Pred_Boxes)


        positive_negative_boxes=tf.where(anchor_state==1,1,0)
        positive_negative_boxes=tf.expand_dims(positive_negative_boxes,axis=-1)
        positive_negative_class_index=box_class_index_array*positive_negative_boxes
        positive_negative_class_index=tf.reshape(tensor=positive_negative_class_index,shape=(-1,1))
        positive_negative_class_index=tf.cast(positive_negative_class_index,dtype=tf.dtypes.float32)
        positive_negative_box_array=tf.concat((box_offset_array,positive_negative_class_index),axis=-1)
        return positive_negative_box_array,anchor_state

    def get_offset_boxes(self):
        """Prepare offset boxes for a batch of input images

        Params:
        Return:
            A batch of offset boxes (Tensor),anchor states of anchor boxes in each image (Tensor)
        """
        batch_size=len(self.B_Boxes)
        offset_boxes=[]
        anchor_states=[]
        for i in range(batch_size):
            offset_box,anchor_state=self.get_positive_negative_boxes(GT_boxes=self.B_Boxes[i],Pred_Boxes=self.predicted_boxes)
            offset_boxes.append(offset_box)
            anchor_states.append(anchor_state)
        offset_boxes_stacked=tf.stack(offset_boxes,axis=0)
        anchor_state_stacked=tf.stack(anchor_states,axis=0)
        anchor_state_stacked=tf.reshape(anchor_state_stacked,shape=[anchor_state_stacked.shape[0],anchor_state_stacked.shape[1]])

        return tf.cast(offset_boxes_stacked,dtype=tf.dtypes.float32),anchor_state_stacked
