import tensorflow as tf
from Utils.utils import Resize_Boxes
from Primary.DataPrep.class_dict import class_dict

class VOC(object):
    def __init__(self,json_data,IMG_WIDTH,IMG_HEIGHT):
        self.json_data=json_data['annotation']
        self.IMG_WIDTH=IMG_WIDTH
        self.IMG_HEIGHT=IMG_HEIGHT
    def width(self):
        return int(self.json_data['size']['width'])
    def height(self):
        return int(self.json_data['size']['height'])
    def obj_to_gt(self):
        obj=self.json_data['object']
        x_center=[]
        y_center=[]
        w_list=[]
        h_list=[]
        class_id=[]
        if isinstance(obj,list):
            for i in range(len(obj)):
                class_id.append(class_dict[obj[i]['name']])
                xmax = int(obj[i]['bndbox']['xmax'])
                xmin = int(obj[i]['bndbox']['xmin'])
                ymax = int(obj[i]['bndbox']['ymax'])
                ymin = int(obj[i]['bndbox']['ymin'])

                w = xmax - xmin
                h = ymax - ymin
                x_c = xmin + (w / 2)
                y_c = ymin + (h / 2)

                x_c,y_c,w,h=Resize_Boxes(self.width(),self.height(),x_c,y_c,w,h)

                x_center.append(x_c / self.IMG_WIDTH)
                y_center.append(y_c / self.IMG_HEIGHT)
                w_list.append(w / self.IMG_WIDTH)
                h_list.append(h / self.IMG_HEIGHT)

            return tf.stack([x_center, y_center, w_list, h_list, class_id], axis=-1)
        else:
            class_id.append(class_dict[obj['name']])
            xmax = int(obj['bndbox']['xmax'])
            xmin = int(obj['bndbox']['xmin'])
            ymax = int(obj['bndbox']['ymax'])
            ymin = int(obj['bndbox']['ymin'])

            w = xmax - xmin
            h = ymax - ymin
            x_c = xmin + (w/2)
            y_c = ymin + (h/2)

            x_c, y_c, w, h = Resize_Boxes(self.width(), self.height(), x_c, y_c, w, h)

            x_center.append(x_c / self.IMG_WIDTH)
            y_center.append(y_c / self.IMG_HEIGHT)
            w_list.append(w / self.IMG_WIDTH)
            h_list.append(h / self.IMG_HEIGHT)

        return tf.stack([x_center, y_center, w_list, h_list, class_id], axis=-1)

