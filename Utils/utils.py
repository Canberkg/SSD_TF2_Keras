import cv2
import tensorflow as tf
from matplotlib import pyplot
from Config import cfg_300


def Resize_Boxes(Image_w,Image_h,x_c,y_c,w_box,h_box):

    ratio_h=cfg_300['IMG_HEIGHT']/Image_h
    ratio_w=cfg_300['IMG_WIDTH']/Image_w
    x_c=x_c*ratio_w
    y_c=y_c*ratio_h
    w_box=w_box*ratio_w
    h_box=h_box*ratio_h

    return x_c,y_c,w_box,h_box

def Visualize_BB(image,BB,scores,color):
    NUM_BB=BB.shape[0]
    for i in range(NUM_BB):
        score="{:.0%}".format(scores[i][0].numpy())
        cv2.rectangle(image,pt1=(int(BB[i,0]-(BB[i,2]/2)),int(BB[i,1]-(BB[i,3]/2))),pt2=(int(BB[i,0]+int(BB[i,2]/2)),int(BB[i,1]+(int(BB[i,3])/2))),color=color,thickness=1)
        cv2.putText(img=image, text=score, org=(int(BB[i,0]-(BB[i,2]/2)), int(BB[i,1]-(BB[i,3]/2)) - 5), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=(0, 255, 255), thickness=1)
    return image

def visualize_feature_maps(ssd_model,img):
    model = ssd_model
    model.summary()
    _,feature_maps=model(img)
    square = 16
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(feature_maps[0, :, :, ix - 1], cmap='jet')
            ix += 1
    # show the figure
    pyplot.show()


def calculate_iou(BB1,BB2):

    bb1_x_min,bb1_y_min,bb1_x_max,bb1_y_max=transform_cordinates(BB1)
    bb2_x_min, bb2_y_min, bb2_x_max, bb2_y_max = transform_cordinates(BB2)
    min_intersection_x=tf.math.maximum(bb1_x_min[:,None,0],bb2_x_min[:,0])
    min_intersection_y=tf.math.maximum(bb1_y_min[:,None,0],bb2_y_min[:,0])
    max_intersection_x=tf.math.minimum(bb1_x_max[:,None,0],bb2_x_max[:,0])
    max_intersection_y=tf.math.minimum(bb1_y_max[:,None,0],bb2_y_max[:,0])
    intersect_w=tf.math.maximum(max_intersection_x-min_intersection_x,0)
    intersect_h=tf.math.maximum(max_intersection_y-min_intersection_y,0)


    Intersect_Area=intersect_h*intersect_w
    BB1_Area=(bb1_x_max-bb1_x_min)*(bb1_y_max-bb1_y_min)
    BB2_Area=(bb2_x_max-bb2_x_min)*(bb2_y_max-bb2_y_min)
    Union_Area=BB1_Area[:,None,0]+BB2_Area[:,0]-Intersect_Area
    IOU=Intersect_Area/Union_Area


    return IOU

def transform_cordinates(BB):
    #from x_c,y_c,w,h to x_min,y_min,x_max,y_max
    x_c,y_c,w,h=BB[:,0],BB[:,1],BB[:,2],BB[:,3]

    x_min = abs(x_c - (0.5 * w))
    y_min = abs(y_c - (0.5 * h))
    x_max = abs(x_c + (0.5 * w))
    y_max = abs(y_c + (0.5 * h))

    x_min=tf.expand_dims(input=x_min,axis=-1)
    y_min = tf.expand_dims(input=y_min, axis=-1)
    x_max = tf.expand_dims(input=x_max, axis=-1)
    y_max = tf.expand_dims(input=y_max, axis=-1)

    return x_min,y_min,x_max,y_max

def nms(BBoxes,BBox_scores):
    chosen_boxes=[]
    chosen_scores = []
    chosen_classes = []
    BBox_scores_sftmx=tf.nn.softmax(BBox_scores)
    Bbox_max_indices=tf.math.argmax(BBox_scores_sftmx,axis=-1)
    Bbox_max_values=tf.math.reduce_max(BBox_scores_sftmx,axis=-1)
    Bbox_object_exist=tf.where(Bbox_max_values>=0.5,Bbox_max_indices,0)
    existing_classes,idx=tf.unique(tf.reshape(Bbox_object_exist,Bbox_object_exist.shape[0]))
    for i in range(existing_classes.shape[0]):
        if existing_classes[i]!=0:
            class_mask=tf.where(Bbox_object_exist == existing_classes[i],True,False)
            class_boxes=tf.gather(BBoxes,tf.where(class_mask == True)[...,0])
            class_scores=tf.boolean_mask(Bbox_max_values,class_mask)
            box_indices=tf.image.non_max_suppression(boxes=class_boxes,scores=class_scores,max_output_size=50,iou_threshold=0.5)
            selected_box=tf.gather(class_boxes,box_indices)
            selected_score=tf.gather(class_scores,box_indices)
            selected_class=existing_classes[i]
            chosen_boxes.append(selected_box)
            chosen_scores.append(selected_score)
            chosen_classes.append(selected_class)
    boxes = tf.concat(values=chosen_boxes, axis=0)
    scores = tf.concat(values=chosen_scores, axis=0)
    classes = tf.concat(values=chosen_classes, axis=0)

    return boxes,scores,classes