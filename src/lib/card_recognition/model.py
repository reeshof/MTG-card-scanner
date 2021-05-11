import utils.util as util

import tensorflow as tf
from keras.models import model_from_json

def effB0_recognition(input_shape,feat_vec_size=128,margin=0.5):
    embedding = embedding_layer(input_shape,feat_vec_size)

    anchor_input = tf.keras.Input(input_shape, name="anchor_input")
    positive_input = tf.keras.Input(input_shape, name="positive_input")
    negative_input = tf.keras.Input(input_shape, name="negative_input")

    encoded_a = embedding(anchor_input)
    encoded_p = embedding(positive_input)
    encoded_n = embedding(negative_input)

    loss_layer = TripletLossLayer(alpha=margin,name='triplet_loss_layer')([encoded_a,encoded_p,encoded_n])
    model_train = tf.keras.Model(inputs=[anchor_input,positive_input,negative_input],outputs=loss_layer)

    anchor_input_pred = tf.keras.Input(input_shape, name="anchor_input_pred")
    encoded_pred = embedding(anchor_input_pred)
    model_predict = tf.keras.Model(inputs=anchor_input_pred,outputs=encoded_pred)

    return [model_train,model_predict]

def embedding_layer(input_shape,feat_vec_size,name=""):
    batch = 1

    embedded_input = tf.keras.Input(input_shape, name="embedded_input"+name, batch_size=batch)

    effB0 = tf.keras.applications.EfficientNetB0(include_top=False,weights="imagenet",input_shape=input_shape)

    for layer in effB0.layers:
        layer.trainable = False

    initializer = tf.keras.initializers.RandomNormal(stddev=0.1)

    output = tf.keras.layers.GlobalAveragePooling2D()(effB0.output)
    output = tf.keras.layers.Dense(feat_vec_size,kernel_initializer = initializer,name="dense_output_layer_1"+name)(output)#128
    output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x,axis=-1))(output)

    embedding = tf.keras.Model(effB0.input, output, name="Embedding"+name)

    return embedding

class TripletLossLayer(tf.keras.layers.Layer):
    #https://medium.com/@crimy/one-shot-learning-siamese-networks-and-triplet-loss-with-keras-2885ed022352
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = tf.keras.backend.sum(tf.keras.backend.square(anchor-positive), axis=-1)
        n_dist = tf.keras.backend.sum(tf.keras.backend.square(anchor-negative), axis=-1)
        return tf.keras.backend.sum(tf.keras.backend.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss
