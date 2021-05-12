import utils.util as util

import tensorflow as tf
import numpy as np
from keras.models import model_from_json

def getEffB0Model(inputshape):
	input = tf.keras.Input(shape=inputshape)
	effB0 = tf.keras.applications.EfficientNetB0(include_top=False,weights="imagenet",input_tensor=input)

	for layer in effB0.layers:
		layer.trainable = False

	upsampled = effB0.output

	#transposed layer doubles the dimensionality.
	initializer = tf.keras.initializers.RandomNormal(stddev=0.1)

	trans = tf.keras.layers.Conv2DTranspose(16,3,2,'same', kernel_initializer = initializer )
	upsampled = trans(upsampled)
	trans = tf.keras.layers.Conv2DTranspose(16,3,2,'same',kernel_initializer = initializer )
	upsampled = trans(upsampled)
	trans = tf.keras.layers.Conv2DTranspose(8,3,2,'same', kernel_initializer = initializer )
	upsampled = trans(upsampled)
	trans = tf.keras.layers.Conv2DTranspose(8,3,2,'same', kernel_initializer = initializer )
	upsampled = trans(upsampled)
	trans = tf.keras.layers.Conv2DTranspose(8,3,2,'same', kernel_initializer = initializer )
	upsampled = trans(upsampled)

	#using efficientnetB0, the output dimensionalityis now the same as the input
	outputMap = tf.keras.layers.Conv2D(5, 3, use_bias=True, padding='same', kernel_initializer = initializer )(upsampled)
	outputMap = tf.keras.layers.Activation('relu')(outputMap)
	outputMap = tf.keras.layers.Conv2D(1, 1, use_bias=True, kernel_initializer = initializer)(outputMap)
	outputMap = tf.keras.layers.ReLU(1.0, name='output_heatmap')(outputMap)

	outputAngle = tf.keras.layers.Conv2D(5, 3, use_bias=True, padding='same', kernel_initializer = initializer )(upsampled)
	outputAngle = tf.keras.layers.Activation('relu')(outputAngle)
	outputAngle = tf.keras.layers.Conv2D(1, 1, use_bias=True, kernel_initializer = initializer, name='output_angle',activation = 'tanh')(outputAngle)

	outputwh = tf.keras.layers.Conv2D(5, 3, use_bias=True, padding='same', kernel_initializer = initializer )(upsampled)
	outputwh = tf.keras.layers.Activation('relu')(outputwh)
	outputwh = tf.keras.layers.Conv2D(2, 1, use_bias=True, kernel_initializer = initializer, name='output_wh')(outputwh)

	effB0 = tf.keras.Model(input,[outputMap,outputAngle, outputwh])
	return effB0

#this model is not used, during experiments it did not show better results
def getHourglassModel():
	json_file = open('trained_models/hg_s2_b1_mobile/net_arch_mobile.json', 'r')
	loaded_model_json = json_file.read()
	hourglass = model_from_json(loaded_model_json)
	hourglass.load_weights('trained_models/hg_s2_b1_mobile/mobile_weights_epoch99.h5')

	for layer in hourglass.layers:
		layer.trainable = False

	conv_output = hourglass.get_layer('1_conv_1x1_x1').output

	initializer = tf.keras.initializers.RandomNormal(stddev=0.0000005)

	outputMap = tf.keras.layers.Conv2D(1, 3, use_bias=True, padding='same', kernel_initializer = initializer )(conv_output)
	outputMap = tf.keras.layers.Activation('relu')(outputMap)
	outputMap = tf.keras.layers.Conv2D(1, 1, use_bias=True, kernel_initializer = initializer, name='output_heatmap',activation = 'sigmoid')(outputMap)

	outputAngle = tf.keras.layers.Conv2D(10, 3, use_bias=True, padding='same', kernel_initializer = initializer )(conv_output)
	outputAngle = tf.keras.layers.Activation('relu')(outputAngle)
	outputAngle = tf.keras.layers.Conv2D(2, 1, use_bias=True, kernel_initializer = initializer, name='output_angle',activation = 'tanh' )(outputAngle)

	outputwh = tf.keras.layers.Conv2D(30, 3, use_bias=True, padding='same', kernel_initializer = initializer )(conv_output)
	outputwh = tf.keras.layers.Activation('relu')(outputwh)
	outputwh = tf.keras.layers.Conv2D(2, 1, use_bias=True, kernel_initializer = initializer, name='output_wh')(outputwh)

	hourglass = tf.keras.Model(hourglass.input,[outputMap,outputAngle, outputwh])

	return hourglass

def focalLoss(groundTruth,predicted):
	#see 'objects as points' paper for much information on this loss function
	epsilon = tf.constant(0.000000001)

	positive_indices = tf.math.equal(groundTruth,tf.constant(1.0))#positive loss indices mask
	negative_indices = tf.math.less(groundTruth,tf.constant(1.0))#negative loss indices mask

	negative_weights = tf.math.pow(tf.math.subtract(1.0, groundTruth),tf.constant(4.0))

	loss = tf.constant(0.0)

	pos_log_term = tf.math.log(tf.math.add(predicted,epsilon)) #add epsilon to prevent log of 0
	pos_alpha_term = tf.math.pow(tf.math.subtract(1.0,predicted),tf.constant(2.0))
	positive_loss = tf.math.multiply(pos_log_term,pos_alpha_term)
	positive_loss = tf.boolean_mask(positive_loss,positive_indices)

	neg_log_term = tf.math.subtract(1.0,predicted)
	neg_log_term = tf.math.add(neg_log_term,epsilon) #add epsilon to prevent log of 0
	neg_log_term = tf.math.log(neg_log_term)
	neg_alpha_term = negative_weights
	neg_beta_term = tf.math.pow(predicted,tf.constant(2.0))
	negative_loss = tf.math.multiply(neg_beta_term,neg_alpha_term)
	negative_loss = tf.math.multiply(negative_loss,neg_log_term)
	negative_loss = tf.boolean_mask(negative_loss,negative_indices)

	num_keypoints = tf.cast(tf.math.count_nonzero(positive_indices), tf.float32)

	positive_loss = tf.math.reduce_sum(positive_loss)#tf.math.multiply(,10.0)
	negative_loss = tf.math.reduce_sum(negative_loss)

	#when the number of keypoints is 0 then there were no objects detected in the image
	if num_keypoints == 0.0:
		loss = loss - negative_loss
	else:
		loss = tf.math.subtract(loss, tf.math.divide(tf.math.add(positive_loss, negative_loss), tf.cast(num_keypoints,tf.float32)))

	return loss

def angleLoss(groundtruth, predicted):
	keypoint_indices = tf.math.greater(groundtruth,tf.constant(0.0))

	gt_sin = tf.math.sin(groundtruth)
	loss_sin = tf.math.abs(tf.math.subtract(gt_sin,predicted))
	loss = tf.boolean_mask(loss_sin,keypoint_indices)
	angle_loss = tf.math.reduce_sum(loss)

	num_keypoints = tf.cast(tf.math.count_nonzero(keypoint_indices), tf.float32)

	return tf.math.divide(angle_loss,num_keypoints)

def whLoss(groundtruth, predicted):
	keypoint_indices = tf.math.greater(groundtruth,tf.constant(0.0))

	loss = tf.math.abs(tf.math.subtract(groundtruth,predicted))
	loss = tf.boolean_mask(loss,keypoint_indices)
	loss = tf.math.reduce_sum(loss)

	num_keypoints = tf.cast(tf.math.count_nonzero(keypoint_indices), tf.float32)

	return tf.math.divide(loss,num_keypoints)

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_epochs = [20,50,135]
    if epoch in decay_epochs:
        return lr * decay_rate
    return lr

def trainModelWithGenerator(train_generator,val_generator,model,epochs,name,model_history_path,callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)],
		):
	H =model.fit(x = train_generator,validation_data=val_generator,epochs=epochs,callbacks=callbacks)

	if name is not None:
		util.plotData(H.history['loss'],H.history['val_loss'],'loss',name,model_history_path)
		util.plotData(H.history['output_heatmap_loss'],H.history['val_output_heatmap_loss'],'heatmap_loss',name,model_history_path,ylim=(0,10))
		util.plotData(H.history['output_angle_loss'],H.history['val_output_angle_loss'],'angle_loss',name,model_history_path)
		util.plotData(H.history['output_wh_loss'],H.history['val_output_wh_loss'],'wh_loss',name,model_history_path)

	util.minmaxLoss(H)

	return H
