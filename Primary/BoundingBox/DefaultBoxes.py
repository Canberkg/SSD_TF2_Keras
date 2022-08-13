import tensorflow as tf
import numpy as np
import math

class Feature_Map(object):
    """Feature Map Class

    Attributes :
            FEATURE_MAPS  : List of feature map shapes  (List)
            IMG_WIDTH     : Width of input accepted by the network (Int)
            IMG_HEIGHT    : Height of input accepted by the network (Int)
            SIZES         : List of scales for anchor boxes (List)
    """
    def __init__(self,Feature_Maps,IMG_WIDTH,IMG_HEIGHT,SIZES):
        self.NUM_Feature_Maps=len(Feature_Maps)
        self.SIZES=SIZES
        self.Feature_Maps=Feature_Maps
        self.IMG_WIDTH=IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT

    def get_length(self):
        """Find the total number of feature maps used for prediction
        Return:
            Number of Feature Maps (int)
        """
        return self.NUM_Feature_Maps
    def get_width(self,idx):
        """Find the width of feature map
        Params:
            idx: Index of the specific feature map (int)
        Return:
            Width of the Feature Map (int)
        """
        return self.Feature_Maps[idx][1]
    def get_height(self,idx):
        """Find the height of feature map
        Params:
            idx: Index of the specific feature map (int)
        Return:
            Height of the Feature Map (int)
        """
        return self.Feature_Maps[idx][0]

    def get_scale_min(self,idx):
        """Find the min scale of anchor boxes for specific feature map
        Params:
            idx: Index of the specific feature map (int)
        Return:
            Min scale of anchor boxes on that feature map (int)
        """
        return self.SIZES[idx][0]
    def get_scale_max(self,idx):
        """Find the max scale of anchor boxes for specific feature map
        Params:
            idx: Index of the specific feature map (int)
        Return:
            Max scale of anchor boxes on that feature map (int)
        """
        return self.SIZES[idx][1]

class DefaultBoxes(object):
    """Default Boxes (Anchor Boxes) Class

    Attributes :
            FEATURE_MAPS  : List of feature map shapes  (List)
            IMG_WIDTH     : Width of input accepted by the network (Int)
            IMG_HEIGHT    : Height of input accepted by the network (Int)
            ASPECT_RATIOS : List of aspect ratios for anchor boxes (List)
            SIZES         : List of scales for anchor boxes (List)
    """
    def __init__(self,Feature_Maps,IMG_WIDTH,IMG_HEIGHT,ASPECT_RATIOS,SIZES):

        self.image_width=IMG_WIDTH
        self.image_height=IMG_HEIGHT
        self.aspect_ratios=ASPECT_RATIOS
        self.feature_maps=Feature_Map(Feature_Maps,IMG_WIDTH=IMG_WIDTH,IMG_HEIGHT=IMG_HEIGHT,
                                      SIZES=SIZES)
        self.Num_Feature_Maps=self.feature_maps.get_length()
        self.Offset=0.5

    def create_default_boxes_for_feature_map(self,idx):
        """Generate anchor boxes for a specific feature map
        Params:
            idx: Index of the specific feature map (int)
        Return:
            x_middle: Center x-coordinate of the anchor boxes (Array)
            y_middle: Center y-coordinate of the anchor boxes (Array)
            Db_width: Width of the anchor boxes (Array)
            Db_height: Height of the anchor boxes (Array)
        """
        Fm_width=self.feature_maps.get_width(idx)
        Fm_height=self.feature_maps.get_height(idx)
        s_idx=self.feature_maps.get_scale_min(idx=idx)
        s_idx_plus=self.feature_maps.get_scale_max(idx=idx)
        aspect_ratios=self.aspect_ratios[idx]

        step_x=self.image_width/Fm_width
        step_y=self.image_height/Fm_height

        x_start=step_x*self.Offset
        x_end=(Fm_width-self.Offset)*step_x
        x_center=np.linspace(x_start,x_end,Fm_width)

        y_start = step_y * self.Offset
        y_end = (Fm_height - self.Offset) * step_y
        y_center = np.linspace(y_start, y_end, Fm_height)

        X,Y=np.meshgrid(x_center,y_center)

        X=X.flatten()
        Y=Y.flatten()
        x_middle=X/self.image_width
        y_middle=Y/self.image_height
        x_middle=np.array(x_middle,dtype=np.float32).reshape(Fm_height,Fm_width)
        y_middle=np.array(y_middle,dtype=np.float32).reshape(Fm_height,Fm_width)

        Db_width=[]
        Db_height=[]
        for i in range(len(aspect_ratios)):
            if aspect_ratios[i] == 1.0:
                s_ext=math.sqrt(s_idx*s_idx_plus)
                Db_height.append(s_ext)
                Db_width.append(s_ext)

            Db_height.append(s_idx / math.sqrt(aspect_ratios[i]))
            Db_width.append(s_idx * math.sqrt(aspect_ratios[i]))



        Db_width=np.array(Db_width,dtype=np.float32)/self.image_width
        Db_height=np.array(Db_height,dtype=np.float32)/self.image_height

        return x_middle,y_middle,Db_width,Db_height

    def generate_default_boxes(self):
        """Combine all the generated anchor boxes for each feature map
        Params:
        Return:
            DefaultBoxes_list: Array of total generated anchor boxes (Array)
        """
        DefaultBoxes_list=[]
        for i in range(self.Num_Feature_Maps):

            X_middle,Y_middle,Widht,Height= self.create_default_boxes_for_feature_map(idx=i)
            XY_middle=np.stack((X_middle,Y_middle),axis=-1)
            WH=np.stack((Widht,Height),axis=-1)
            DB_for_Current_Feature_Map=[]
            for k in range(XY_middle.shape[0]):
                for j in range(XY_middle.shape[1]):
                    for l in range(WH.shape[0]):
                        DB_xy=np.concatenate((XY_middle[k,j],WH[l]),axis=0)
                        DB_for_Current_Feature_Map.append(DB_xy)

            DB_for_Current_Feature_Map=np.stack(DB_for_Current_Feature_Map,axis=0)
            DefaultBoxes_list.append(DB_for_Current_Feature_Map)

        DefaultBoxes_list=tf.concat(DefaultBoxes_list,axis=0)

        return DefaultBoxes_list
