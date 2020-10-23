from RetinaFace.retinaface_cov import RetinaFaceCoV
import tensorflow as tf
import keras
from parameters import *

def triplet_loss(y_true, y_pred, alpha = ALPHA):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))    
    return loss

def load_models():
    thresh = 0.8
    mask_thresh = 0.2
    count = 1
    gpuid = 0
    detector = RetinaFaceCoV('./models/cov2/mnet_cov2', 0, -1, 'net3l')
    print("Trained model found")
    print("Loading custom trained model...")
    model = keras.models.load_model('models/facenet_keras.h5', custom_objects={'triplet_loss': triplet_loss})        
    print('Completed')
        
    return detector, model