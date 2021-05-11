import object_detection.train as train
import object_detection.model as model
import object_detection.train_generator as train_generator
import utils.util as util

import os
import cv2 as cv
import tensorflow as tf
import numpy as np
import imgaug.augmenters as iaa
from pathlib import Path

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.compat.v1.disable_eager_execution() #THIS HELPS A LOT FOR PREDICTION TIME

class object_detection_model():
    #assume working directory is 'src'
    weights_path = '../models/weights/'
    train_results_path = '../models/train_history/'

    train_path = '../data/obj_det_training/'
    test_path='../data/obj_det_test/images/'
    output_path='../data/obj_det_test_result/images/'

    def __init__(self, dim=(512,384)):
        self.dim=dim
        self.model = model.getEffB0Model((*self.dim,3))

    def load_generators(self, batch_size=10,batches_per_epoch=15,seed=1,output_stride=1):
        [self.file_names,self.partition,self.annotations] = train.get_dictionaries(0.7, object_detection_model.train_path)

        self.seq = iaa.Sequential([
        	iaa.Invert(0.5),
            iaa.Affine(rotate=(-15,75),scale=(0.8,1.2),translate_percent={"x":(-0.15,0.15), "y": (-0.15,0.15)})#old was -15,
        ])

        self.generator_params = {'dim': self.dim,
              'batch_size': batch_size,
              'batches_per_epoch': batches_per_epoch,
              'seed': seed,
              'output_stride':output_stride,
              'training_path':object_detection_model.train_path+'train_set/'}

        [file_names,partition,annotations] = train.get_dictionaries(0.7,object_detection_model.train_path)
        self.training_generator = train_generator.DataGenerator(self.partition['train_ids'], self.file_names, self.annotations, self.seq, **self.generator_params)
        self.validation_generator = train_generator.DataGenerator(self.partition['val_ids'], self.file_names, self.annotations, self.seq, **self.generator_params)

    def compile(self, learning_rate=0.001):
        self.opt = tf.keras.optimizers.Adam(learning_rate)
        self.callbacks=[tf.keras.callbacks.LearningRateScheduler(model.lr_scheduler, verbose=1)]
        self.loss= {'output_heatmap':model.focalLoss,'output_angle':model.angleLoss,'output_wh':model.whLoss}
        self.loss_weights= {'output_heatmap':1.0,'output_angle':10.0,'output_wh':0.1}
        self.model.compile(optimizer=self.opt, loss=self.loss,loss_weights=self.loss_weights)

    def set_layer(self, layer_name, is_trainable):
        self.model.get_layer(layer_name).trainable=is_trainable

    def train(self, epochs, train_name=None):
        if train_name is not None:
            os.mkdir(object_detection_model.train_results_path+train_name)
        model.trainModelWithGenerator(self.training_generator,self.validation_generator,self.model,epochs,train_name,object_detection_model.train_results_path,self.callbacks)

    def load_weights(self, weight_file):
        self.model.load_weights(object_detection_model.weights_path+weight_file+'.h5',by_name=True,skip_mismatch=True)

    def save_weights(self, weight_file):
        self.model.save_weights(object_detection_model.weights_path+weight_file+".h5",overwrite=False)

    def get_predictions(self, img):
        img = cv.resize(img,tuple(np.flip(self.dim)))

        predicted = self.model.predict(img.reshape(1,*self.dim,3))

        return (img,predicted)

    def get_regions(self,img,predicted):
        detected_regions = get_objects(img,predicted,0,self.dim)
        return detected_regions

    def predict_image(self,img,filename):
        img = cv.resize(img,tuple(np.flip(self.dim)))

        predicted = self.model.predict(img.reshape(1,*self.dim,3))

        detected_regions = get_objects(img,predicted,0,self.dim)

        for i,region in enumerate(detected_regions):
            cv.imwrite(object_detection_model.output_path + Path(filename).stem + str(i)+'.png', region)

        return detected_regions

    def predict_directory(self):
        for filename in os.listdir(object_detection_model.test_path):
            img = cv.imread(object_detection_model.test_path+filename)
            self.predict_image(img,filename)


def get_objects(img,predicted,num_keypoints,dim):
    widthheight = np.split(predicted[2],2,-1)

    local_max_indices= util.find_local_maximum(predicted[0].reshape(*dim))

    cropped_images = []
    threshold=0.2
    num_regions =0
    for i in np.arange(0,len(local_max_indices)):
        if local_max_indices[i][0] > threshold:
            num_regions+=1
            local_max_index = local_max_indices[i][1]
            object_centerpoint= (0,local_max_index[0],local_max_index[1],0)

            width = widthheight[0][object_centerpoint]
            height = widthheight[1][object_centerpoint]

            topleftpoint = np.flip([object_centerpoint[1]- (height / 2.0), object_centerpoint[2] - (width/2.0)])

            predicted_orientation = util.compute_orientation_vec(predicted[1][object_centerpoint])
            origin_point = [object_centerpoint[1],object_centerpoint[2]]

            destination_point = np.add(origin_point,np.multiply(util.unit_vector(predicted_orientation),height/2))

            cropped_image = util.extract_roi(img,[object_centerpoint[1],object_centerpoint[2]],width,height,np.arctan2(predicted_orientation[1],-predicted_orientation[0]))
            cropped_images.append(cropped_image)

    #print("found number of regions: ",num_regions)
    return cropped_images

def get_boundingboxes(predicted, dim):
    widthheight = np.split(predicted[2],2,-1)
    local_max_indices= util.find_local_maximum(predicted[0].reshape(*dim))

    threshold=0.2
    num_regions =0
    bbs = [] #[(topleft_x,topright_x),(botright_x,botright_y),(og_x,og_y),(dest_x,dest_y)]
    for i in np.arange(0,len(local_max_indices)):
        if local_max_indices[i][0] > threshold:
            num_regions+=1
            local_max_index = local_max_indices[i][1]
            object_centerpoint= (0,local_max_index[0],local_max_index[1],0)

            width = widthheight[0][object_centerpoint]
            height = widthheight[1][object_centerpoint]

            topleftpoint = np.flip([object_centerpoint[1]- (height / 2.0), object_centerpoint[2] - (width/2.0)])
            botrightpoint = np.flip([object_centerpoint[1] + (height / 2.0), object_centerpoint[2] + (width/2.0)])

            predicted_orientation = util.compute_orientation_vec(predicted[1][object_centerpoint])
            origin_point = [object_centerpoint[1],object_centerpoint[2]]
            destination_point = np.add(origin_point,np.multiply(util.unit_vector(predicted_orientation),height/2))

            bbs.append((topleftpoint,botrightpoint,origin_point,destination_point))

    return bbs
