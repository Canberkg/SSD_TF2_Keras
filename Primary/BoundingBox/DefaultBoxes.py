import tensorflow as tf
import numpy as np
import math

class Feature_Map(object):
    def __init__(self,Feature_Maps,IMG_WIDTH,IMG_HEIGHT,MIN_SCALE,MAX_SCALE):
        self.NUM_Feature_Maps=len(Feature_Maps)
        self.min_scale=MIN_SCALE
        self.max_scale=MAX_SCALE
        self.Feature_Maps=Feature_Maps
        self.IMG_WIDTH=IMG_WIDTH
        self.IMG_HEIGHT = IMG_HEIGHT

    def get_length(self):
        return self.NUM_Feature_Maps
    def get_width(self,idx):
        return self.Feature_Maps[idx][1]
    def get_height(self,idx):
        return self.Feature_Maps[idx][0]
    def get_downsample_ratio(self,idx):
        width_ratio=self.IMG_WIDTH/self.get_width(idx)
        height_ratio=self.IMG_HEIGHT/self.get_height(idx)
        if width_ratio !=height_ratio:
            raise ValueError("Downsampling ratio of width and height should be equal!")
        else:
            return width_ratio
    def get_scale(self,idx):
        return self.min_scale+(((self.max_scale-self.min_scale)*idx)/(self.get_length()-1))

class DefaultBoxes(object):
    def __init__(self,Feature_Maps,IMG_WIDTH,IMG_HEIGHT,ASPECT_RATIOS,MIN_SCALE,MAX_SCALE):
        self.image_width=IMG_WIDTH
        self.image_height=IMG_HEIGHT
        self.aspect_ratios=ASPECT_RATIOS
        self.feature_maps=Feature_Map(Feature_Maps,IMG_WIDTH=IMG_WIDTH,IMG_HEIGHT=IMG_HEIGHT,
                                      MIN_SCALE=MIN_SCALE,MAX_SCALE=MAX_SCALE)
        self.Num_Feature_Maps=self.feature_maps.get_length()
        self.Offset=0.5

    def create_default_boxes_for_feature_map(self,idx):
        #Properties
        Fm_width=self.feature_maps.get_width(idx)
        Fm_height=self.feature_maps.get_height(idx)
        s_idx=self.feature_maps.get_scale(idx)*self.feature_maps.get_downsample_ratio(idx)
        s_idx_plus=self.feature_maps.get_scale((idx+1)%self.Num_Feature_Maps)*self.feature_maps.get_downsample_ratio((idx+1)%self.Num_Feature_Maps)
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
        DefaultBoxes_list=[]
        for i in range(self.Num_Feature_Maps):
            #Calculate Cebter points and Width and Height of each deafult box for a specific feature map
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
