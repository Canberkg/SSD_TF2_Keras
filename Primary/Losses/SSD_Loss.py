import tensorflow as tf
from keras import backend as K
import numpy as np



def _smooth_l1(y_true, y_pred, anchor_state):

    sigma = 3.0
    sigma_squared = sigma ** 2

    regression = y_pred
    regression_target = y_true
    anchor_state = anchor_state


    indices = tf.where(tf.keras.backend.equal(anchor_state, 1))
    regression = tf.gather_nd(regression, indices)
    regression_target = tf.gather_nd(regression_target, indices)


    regression_diff = regression - regression_target
    regression_diff = tf.keras.backend.abs(regression_diff)
    regression_loss = tf.where(
        tf.keras.backend.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.keras.backend.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )


    normalizer = tf.keras.backend.maximum(1, tf.keras.backend.shape(indices)[0])
    normalizer = tf.keras.backend.cast(normalizer, dtype=tf.keras.backend.floatx())
    return tf.keras.backend.sum(regression_loss) / normalizer


def ConfidenceLoss(y_true, y_pred,NUM_CLASSES):
    GT_class = y_true[..., -1]
    Pred_class = y_pred[..., :NUM_CLASSES]

    Gt_one_hot = tf.one_hot(tf.cast(GT_class, dtype=tf.dtypes.int32), depth=NUM_CLASSES)

    pos_box = tf.where(GT_class > 0.0, 1.0, 0.0)
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    cross_e = cce(Gt_one_hot, Pred_class)
    num_pos = tf.cast(tf.math.count_nonzero(input=pos_box, axis=-1), dtype=tf.dtypes.float32)
    num_neg = 3.0 * num_pos

    neg_cross = tf.where(tf.equal(x=pos_box, y=0.0), cross_e, 0.0)
    sorted = tf.cast(tf.argsort(neg_cross, direction='DESCENDING', axis=-1), tf.int32)
    rank = tf.cast(tf.argsort(sorted, axis=-1), tf.int32)
    num_neg = tf.cast(num_neg, dtype=tf.int32)
    neg_loss = tf.where(rank < tf.expand_dims(num_neg, axis=1), neg_cross, 0.0)

    pos_loss = tf.where(tf.equal(x=pos_box, y=1.0), cross_e, 0.0)

    conf_loss = tf.reduce_sum(pos_loss + neg_loss, axis=-1) / tf.maximum(1.0, num_pos)
    return tf.math.reduce_sum(conf_loss)


def SSDLoss(y_true, y_pred, anchor_state,NUM_CLASSES):

    y_true_cord = y_true[:, :, :4]
    y_pred_cord = y_pred[:, :, NUM_CLASSES:]

    loc_loss = _smooth_l1(y_true=y_true_cord, y_pred=y_pred_cord, anchor_state=anchor_state)
    conf_loss = ConfidenceLoss(y_true=y_true, y_pred=y_pred,NUM_CLASSES=NUM_CLASSES)

    loss = loc_loss + conf_loss
    # print(f"conf_loss : {conf_loss} , loc_loss : {loc_loss}")
    return loss, loc_loss, conf_loss