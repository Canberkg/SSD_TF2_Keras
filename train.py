import os
import time
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from Detection.SSD300 import SSD300

from Primary.Losses.SSD_Loss import SSDLoss
from Primary.DataPrep.datagen import data_gen
from Primary.BoundingBox.GroundTruth import GroundTruth
from Config import cfg_300


def ssd_training_300(root_dir_train,root_dir_valid,_root_dir_train_jsons,root_dir_valid_jsons):
    ## Initialize hyperparameters from Cfg file

    BATCH_SIZE    = cfg_300['BATCH_SIZE']
    EPOCH         = cfg_300['EPOCH']
    IMG_WIDTH     = cfg_300['IMG_WIDTH']
    IMG_HEIGHT    = cfg_300['IMG_HEIGHT']
    NUM_CLASSES   = cfg_300['NUM_CLASS']
    LR            = cfg_300['LR']
    LR_SCHEDULER  = cfg_300['LR_SCHEDULER']
    DECAY_STEPS   = cfg_300['DECAY_STEPS']
    DECAY_RATE    = cfg_300['DECAY_RATE']
    OPTIMIZER     = cfg_300['OPTIMIZER']
    ASPECT_RATIOS = cfg_300['ASPECT_RATIOS']
    MIN_SCALE     = cfg_300['MIN_SCALE']
    MAX_SCALE     = cfg_300['MAX_SCALE']
    IOU_THRESHOLD = cfg_300['IOU_THRESHOLD']
    FEATURE_MAPS  = cfg_300['FEATURE_MAPS']
    SHUFFLE       = cfg_300['SHUFFLE']
    SAVE_DIR      = cfg_300['SAVE_DIR']
    MODEL_NAME    = cfg_300['MODEL_NAME']
    CKPT_DIR      = cfg_300['CKPT_DIR']


    train_generator = data_gen(Img_Path=root_dir_train,Label_Path=root_dir_train_jsons,
                               Img_Width=IMG_WIDTH,Img_Height=IMG_HEIGHT,Batch_Size=BATCH_SIZE,
                               Num_Classes=NUM_CLASSES,Shuffle=SHUFFLE)
    valid_generator = data_gen(Img_Path=root_dir_valid, Label_Path=root_dir_valid_jsons,
                               Img_Width=IMG_WIDTH, Img_Height=IMG_HEIGHT, Batch_Size=BATCH_SIZE,
                               Num_Classes=NUM_CLASSES, Shuffle=SHUFFLE)

    ssd_model = SSD300(ASPECT_RATIOS=ASPECT_RATIOS,BATCH_SIZE=BATCH_SIZE,NUM_CLASS=NUM_CLASSES)

    if  LR_SCHEDULER == True:
        lr=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=LR,decay_steps=DECAY_STEPS,decay_rate=DECAY_RATE)
    else:
        lr=LR

    if OPTIMIZER.upper() == 'ADAM' :
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif OPTIMIZER.upper() == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9)
    else:
        raise Exception("Need to specify an existing optimizer")

    # metrics
    training_loss = tf.metrics.Mean()
    validation_loss = tf.metrics.Mean()
    metrics_names = ["training loss","validation loss"]

    best = 9999
    patience = 30
    wait = 0

    checkpoint=tf.train.Checkpoint(optimizer=optimizer,model=ssd_model)
    ckpt_manager=tf.train.CheckpointManager(checkpoint=checkpoint,directory=os.path.join(SAVE_DIR,CKPT_DIR),max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print("checkpoint named as {} is restored ".format(ckpt_manager.latest_checkpoint))
    else:
        print("Model will be initialized from scratch as there are no checkpoints !! ")

    train_loss = []
    valid_loss = []
    epochs = []


    def train_step(Image_Batch, GT_boxes):
        with tf.GradientTape() as tape:
            ssd_pred = ssd_model(Image_Batch, training=True)
            labeled_boxes = GroundTruth(Boxes=GT_boxes, FEATURE_MAPS=FEATURE_MAPS,IMG_WIDTH=IMG_WIDTH,
                                        IMG_HEIGHT=IMG_HEIGHT,IOU_THRESHOLD=IOU_THRESHOLD,ASPECT_RATIOS=ASPECT_RATIOS,
                                        MIN_SCALE=MIN_SCALE,MAX_SCALE=MAX_SCALE)
            gt_offsets, anchor_state = labeled_boxes.get_offset_boxes()
            total_loss, loc_loss, conf_loss = SSDLoss(y_true=gt_offsets, y_pred=ssd_pred,anchor_state=anchor_state,
                                                      NUM_CLASSES=NUM_CLASSES)

        grads = tape.gradient(total_loss, ssd_model.trainable_weights)
        optimizer.apply_gradients(zip(grads, ssd_model.trainable_weights))
        return total_loss


    def test_step(Image_Batch,GT_boxes):
        ssd_pred = ssd_model(Image_Batch, training=False)
        labeled_boxes = GroundTruth(Boxes=GT_boxes, FEATURE_MAPS=FEATURE_MAPS, IMG_WIDTH=IMG_WIDTH,
                                    IMG_HEIGHT=IMG_HEIGHT, IOU_THRESHOLD=IOU_THRESHOLD, ASPECT_RATIOS=ASPECT_RATIOS,
                                    MIN_SCALE=MIN_SCALE, MAX_SCALE=MAX_SCALE)
        gt_offsets, anchor_state = labeled_boxes.get_offset_boxes()
        total_loss, loc_loss, conf_loss = SSDLoss(y_true=gt_offsets, y_pred=ssd_pred, anchor_state=anchor_state,
                                                  NUM_CLASSES=NUM_CLASSES)
        return total_loss


    for epoch in range(EPOCH):
        pb_i = tf.keras.utils.Progbar(train_generator.num_images(), stateful_metrics=metrics_names)
        batch_training_loss=[]
        batch_validation_loss = []
        print("\n Epoch : {}/{} -".format(epoch, EPOCH))
        for batch_index in range(train_generator.__len__()):
            Images,GT_boxes = train_generator.__getitem__(index=batch_index)
            batch_loss=train_step(Image_Batch=Images, GT_boxes=GT_boxes)
            batch_training_loss.append(batch_loss)
            pb_i.update((batch_index+1) * BATCH_SIZE, values=[('training loss', batch_loss)])
        training_loss.update_state(values=batch_training_loss)
        temp_tl = training_loss.result()
        pb_i.update(current=train_generator.num_images(),values=[('training loss', training_loss.result())], finalize=True)

        for batch_index in range(valid_generator.__len__()):
            Images, GT_boxes = valid_generator.__getitem__(index=batch_index)
            batch_valid_loss=test_step(Image_Batch=Images,GT_boxes=GT_boxes)
            batch_validation_loss.append(batch_valid_loss)
        validation_loss.update_state(values=batch_validation_loss)
        mean_val=validation_loss.result()
        pb_i.update(current=train_generator.num_images(),values=[('training loss', training_loss.result()),('validation loss',mean_val)],finalize=True)


        training_loss.reset_states()
        validation_loss.reset_states()

        train_loss.append(temp_tl)
        valid_loss.append(mean_val)
        epochs.append(epoch)

        train_generator.on_epoch_end()
        valid_generator.on_epoch_end()
        ckpt_manager.save()
        wait += 1
        if temp_tl < best:
            wait = 0
            best = temp_tl
        if wait >= patience:
            break

    ssd_model.save_weights(filepath=os.path.join(SAVE_DIR,"{}.h5".format(MODEL_NAME)),overwrite=True, save_format='h5', options=None)
    training_loss_graph = np.stack((epochs, train_loss), axis=-1)
    validation_loss_graph=np.stack((epochs,valid_loss),axis=-1)
    plt.plot(training_loss_graph[..., 0], training_loss_graph[..., 1], 'r', label='Training loss')
    plt.plot(validation_loss_graph[..., 0], validation_loss_graph[..., 1], 'g', label='Validation loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training loss','Validation Loss'])
    plt.show()


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    root_dir_train       = cfg_300['TRAIN_IMG']
    root_dir_valid       = cfg_300['VALID_IMG']
    root_dir_train_jsons = cfg_300['TRAIN_LABEL']
    root_dir_valid_jsons = cfg_300['VALID_LABEL']

    ssd_training_300(root_dir_train,root_dir_valid,root_dir_train_jsons,root_dir_valid_jsons)