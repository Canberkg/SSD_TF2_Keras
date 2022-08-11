import os
import json
import tensorflow as tf

from cv2 import cv2
from Utils.utils import Visualize_BB
from Primary.DataPrep.VOC import VOC
from Config import cfg_300

from Primary.BoundingBox.GroundTruth import GroundTruth
def total_matched():

        root_dir_Jsons = "D:\\PersonalResearch\\Projects\SSD\\Dataset\\annotations_json"
        json_list = os.listdir(root_dir_Jsons)
        total_pos = 0
        for i in range(len(json_list)):
            if i % 100 == 0:
                print('gt-{} is processing!'.format(i))
            f = open(os.path.join(root_dir_Jsons, json_list[i]))
            data = json.load(f)
            voc = VOC(json_data=data, IMG_WIDTH=300, IMG_HEIGHT=300)
            gt_boxes = voc.obj_to_gt()
            if gt_boxes.shape[0] > 0:
                gt = GroundTruth(Boxes=gt_boxes, FEATURE_MAPS=FEATURE_MAPS)
                _, anchor_state, iou_class_indexes = gt.get_positive_negative_boxes(GT_boxes=gt.B_Boxes,
                                                                                    Pred_Boxes=gt.predicted_boxes)
                pos_box = tf.where(anchor_state == 1, 1, 0)
                num_matched_box = tf.unique(((iou_class_indexes + 1)[:, 0] * pos_box))[0]
                any_matched = tf.where(tf.unique(((iou_class_indexes + 1)[:, 0] * pos_box))[0] > 0).shape[0]
                if any_matched > 0:
                    total_pos = total_pos + ((num_matched_box.shape[0] - 1))
            f.close()

        return total_pos
if __name__=="__main__":
    #total=total_matched()
    #print(total)


    IMG_WIDTH     = cfg_300['IMG_WIDTH']
    IMG_HEIGHT    = cfg_300['IMG_HEIGHT']
    ASPECT_RATIOS = cfg_300['ASPECT_RATIOS']
    SIZES         = cfg_300['SIZES']
    IOU_THRESHOLD = cfg_300['IOU_THRESHOLD']
    FEATURE_MAPS  = cfg_300['FEATURE_MAPS']


    root_dir_real_train = cfg_300['TRAIN_IMG']
    root_dir_Jsons = cfg_300['TRAIN_LABEL']
    json_list=os.listdir(root_dir_Jsons)
    label_arr = ("2008_000217.jpg").split('.')[0]
    label = '{}.json'.format(label_arr)
    json_id=json_list.index(label)
    f = open(os.path.join(root_dir_Jsons, json_list[json_id]))
    data = json.load(f)
    voc = VOC(json_data=data, IMG_WIDTH=300, IMG_HEIGHT=300)
    GT_boxes = voc.obj_to_gt()

    img = "2008_000217.jpg"
    print(img)
    Img_List = []
    GT_List = []


    gt=GroundTruth(Boxes=GT_boxes, FEATURE_MAPS=FEATURE_MAPS,IMG_WIDTH=IMG_WIDTH,
                    IMG_HEIGHT=IMG_HEIGHT,IOU_THRESHOLD=IOU_THRESHOLD,
                    ASPECT_RATIOS=ASPECT_RATIOS,SIZES=SIZES)
    offset,anchor_state=gt.get_positive_negative_boxes(GT_boxes=gt.B_Boxes,Pred_Boxes=gt.predicted_boxes)
    print(tf.math.reduce_sum(tf.where(tf.equal(anchor_state,1),1,0)))
    matched_boxes=tf.gather(params=gt.predicted_boxes,indices=tf.where(tf.equal(anchor_state,1))[:,0])
    scores_db=tf.zeros_like(matched_boxes)
    numpy_mb = matched_boxes.numpy()
    numpy_mb[..., 0] = numpy_mb[..., 0] * 300
    numpy_mb[..., 2] = numpy_mb[..., 2] * 300
    numpy_mb[..., 1] = numpy_mb[..., 1] * 300
    numpy_mb[..., 3] = numpy_mb[..., 3] * 300
    image = cv2.imread(filename=os.path.join(root_dir_real_train, img))
    image = cv2.resize(image, dsize=(300, 300))
    # image=cv2.resize(image,dsize=(300,300))
    image = Visualize_BB(image,numpy_mb, scores=scores_db,color=(255, 0, 0))
    scores_gt=tf.zeros_like(GT_boxes)
    numpy_gt=GT_boxes.numpy()
    numpy_gt[..., 0] = numpy_gt[..., 0] * 300
    numpy_gt[..., 2] = numpy_gt[..., 2] * 300
    numpy_gt[..., 1] = numpy_gt[..., 1] * 300
    numpy_gt[..., 3] = numpy_gt[..., 3] * 300
    image = Visualize_BB(image, numpy_gt[...,:4],scores=scores_gt, color=(0, 255, 0))
    cv2.imshow("image", image)
    cv2.waitKey()
