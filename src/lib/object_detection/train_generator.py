import utils.util as util
import object_detection.train as train

import numpy as np
import tensorflow as tf
import cv2 as cv
import imgaug as ia
import imgaug.augmenters as iaa

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, file_names, annotations, seq, batches_per_epoch, training_path, output_stride =1,batch_size=32, dim=(512,384), n_channels=3,shuffle=True,seed=1):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.file_names = file_names
        self.annotations = annotations
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.batches_per_epoch = batches_per_epoch
        self.seq = seq
        self.output_stride = output_stride
        self.rng = np.random.default_rng(seed)
        self.training_path = training_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        self.rng.shuffle(self.list_IDs)#shuffle on each batch
        IDs = self.list_IDs[:self.batch_size]

        # Generate data
        X, y = self.__data_generation(IDs)

        return X, y

    def on_epoch_end(self):
        'Currently not doing anything, instead the train images are shuffled on each batch'

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        output_dim = np.divide(list(self.dim),self.output_stride).astype(int)

        images = np.empty((self.batch_size,*self.dim,self.n_channels),dtype='uint8')
        heatmaps = np.zeros((self.batch_size,*output_dim,1),dtype=np.float32)
        anglemaps = np.zeros((self.batch_size,*output_dim,1),dtype=np.float32)
        whmaps = np.zeros((self.batch_size,*output_dim,2),dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            img = cv.imread(self.training_path + self.file_names[ID])
            img_height = img.shape[0]
            img_width =  img.shape[1]

            img = cv.resize(img,(self.dim[1],self.dim[0]))

            resfactor = [0,0]
            resfactor[0] = self.dim[0] / img_height #/ self.output_stride
            resfactor[1] = self.dim[1] / img_width #/ self.output_stride
            segment_mask = np.tile(np.array(resfactor),4)

            bbs = np.multiply(self.annotations[ID],segment_mask)

            polygons = util.bbs_to_polygons(bbs)
            img,trans_polygons = self.seq(images=[img],polygons=[polygons])
            img = img[0]
            images[i] = img
            trans_bbs = util.polygons_to_bbs(trans_polygons[0])

            segment_mask = np.tile(np.array([1.0/self.output_stride,1.0/self.output_stride]),4)

            for segment in trans_bbs:
                segment = np.multiply(segment,segment_mask)
                centerpoint = np.flip(np.round(util.computerCenterPointPolygon(segment),0).astype(int))

                if(util.point_outside_bounds(centerpoint,output_dim)):
                    #print(centerpoint)
                    break

                object_size = util.polygonbox_width_height(segment)
                gaussian_width = util.gaussian_radius(object_size)

                whmaps[i].itemset((centerpoint[0],centerpoint[1],0), object_size[0])
                whmaps[i].itemset((centerpoint[0],centerpoint[1],1), object_size[1])

                angle = util.polygonbox_angle(segment)

                anglemaps[i].itemset((centerpoint[0],centerpoint[1],0), angle +0.00000001)

                heatmaps[i] = train.addBoxToHeatMap(heatmaps[i], centerpoint, gaussian_width,angle.astype(int)).reshape(*output_dim,1)

        return images, {'output_heatmap':heatmaps,'output_angle':anglemaps,'output_wh':whmaps}
