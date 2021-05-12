import utils.util as util

import numpy as np
import tensorflow as tf
import cv2 as cv

#https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, file_names, seq, model,image_database_path, batch_size=4, dim=(512,384), n_channels=3,shuffle=True,seed=1,feat_vec_size=128):
        'Initialization'
        self.dim = dim

        self.batch_size = batch_size
        self.mine_size = 128
        self.mine_batch_ratio = int(self.mine_size/self.batch_size)

        self.list_IDs = list_IDs
        self.file_names = file_names
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.model = model
        self.seq = seq
        self.image_database_path = image_database_path
        self.rng = np.random.default_rng(seed)

        self.feat_vec_size = feat_vec_size
        self.feat_vecs= np.empty((len(self.list_IDs),self.feat_vec_size),dtype=np.float32)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(int(len(self.list_IDs) / self.mine_size) * self.mine_batch_ratio)

    def __getitem__(self, index):
        'Generate one batch of data'
        batch_index = index * self.batch_size
        mine_index = int(index/(self.mine_batch_ratio)) * self.mine_size

        anchor_IDs = self.list_IDs[batch_index:
            batch_index + self.batch_size]

        negative_IDs = np.empty(self.batch_size)

        for i,feat_vec in enumerate(self.feat_vecs[batch_index:batch_index + self.batch_size]):
            f = lambda x: np.linalg.norm(feat_vec-x)
            min_distance=100000
            min_index=-1

            pos_index = batch_index + i

            for j,feat_vec_2 in enumerate(self.feat_vecs[mine_index: mine_index + self.mine_size ]):
                neg_index = mine_index + j
                if pos_index != neg_index:

                    dist = f(feat_vec_2)

                    if dist < min_distance:
                        min_distance = dist
                        min_index = j

            negative_IDs[i] = self.list_IDs[mine_index+min_index]

        [anchor,pos,neg] = self.__data_generation(anchor_IDs, negative_IDs)

        return [anchor,pos,neg]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.rng.shuffle(self.list_IDs)

        for i in np.arange(0,self.__len__()*self.batch_size,self.mine_size):
            images = np.empty((self.mine_size,*self.dim,self.n_channels),dtype='uint8')
            for j,id in enumerate(self.list_IDs[i:i+self.mine_size]):
                img = cv.imread(self.image_database_path + self.file_names[id])
                img = cv.resize(img,(self.dim[1],self.dim[0]))

                images[j] = img

            predictions = self.model.predict_on_batch(images)

            for j in np.arange(0,self.mine_size):
                self.feat_vecs[i+j] = predictions[j]
            print(i,end='\r')

    def __data_generation(self, anchor_IDs, negative_IDs):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        output_dim = np.divide(list(self.dim),1).astype(int)

        anchors = np.empty((self.batch_size,*self.dim,self.n_channels),dtype='uint8')
        positive_samples = np.empty((self.batch_size,*output_dim,self.n_channels),dtype='uint8')
        negative_samples = np.empty((self.batch_size,*output_dim,self.n_channels),dtype='uint8')

        for i, (anchor_ID,negative_ID) in enumerate(zip(anchor_IDs,negative_IDs)):
            anchor_img = cv.imread(self.image_database_path + self.file_names[anchor_ID])
            neg_img = cv.imread(self.image_database_path + self.file_names[negative_ID])

            anchor_img = cv.resize(anchor_img,(self.dim[1],self.dim[0]))
            neg_img = cv.resize(neg_img,(self.dim[1],self.dim[0]))

            pos_img = self.seq(images=[anchor_img])
            pos_img = pos_img[0]

            anchors[i] = anchor_img
            positive_samples[i] = pos_img
            negative_samples[i] = neg_img

        return [anchors,positive_samples,negative_samples]
