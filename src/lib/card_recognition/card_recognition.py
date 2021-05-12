import card_recognition.train as train
import card_recognition.train_generator as train_generator
import card_recognition.model as model
import utils.util as util

import imgaug.augmenters as iaa
import tensorflow as tf
import cv2 as cv
import numpy as np
import json
import time

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.compat.v1.disable_eager_execution() #THIS HELPED A LOT FOR PREDICTION TIME

class card_recognition_model():
    _weights_path = '../models/weights/'
    _test_images = '../data/img_rec_test/'
    _train_images= '../images/'#the scryfall card images ~55000 images
    _feature_files = '../data/recognition_features/'
    _output_examples = '../../demoimages/'

    def __init__(self, dim=(204,146),feat_vec_size=256):
        [self.partition,self.file_names] = train.get_dictionaries(card_recognition_model._train_images+'scryfall/',0.9)
        self.dim=dim
        self.loss= None
        self.feat_vec_size = feat_vec_size

        [self.model,self.model_pred]  = model.effB0_recognition((*self.dim,3),self.feat_vec_size)

        layers_top = ['block7a_project_conv', 'top_conv', 'block7a_expand_conv', 'block6d_project_conv', 'block6c_project_conv', 'block6b_project_conv', 'block6a_project_conv', 'block6b_expand_conv']
        layers_mid = ['block6a_expand_conv','block5c_project_conv','block5c_expand_conv','block5b_dwconv','block5b_expand_conv','block5a_project_conv']
        layers_bot = ['block5a_expand_conv','block4c_project_conv','block4c_expand_conv','block4b_project_conv','block4b_expand_conv','block4a_project_conv','block4a_expand_conv']

        for layer in layers_top:
            self.model.layers[3].get_layer(layer).trainable = True
        for layer in layers_mid:
            self.model.layers[3].get_layer(layer).trainable = True

    def load_generators(self,num_train=None,num_val=None,batch_size=32,seed=1):
        self.seq = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0.0, 2.0)),
            iaa.GammaContrast((0.5, 2.0)),
        	iaa.MultiplyBrightness((0.6, 1.4)),
            iaa.Affine(rotate=(-15,15),scale=(0.9,1.1),translate_percent={"x":(-0.05,0.05), "y": (-0.075,0.075)})#old was -15,35
        ])

        self.generator_params = {'dim': self.dim,
              'batch_size': batch_size,
              'seed': seed,
              'model': self.model_pred,
              'image_database_path': card_recognition_model._train_images+'scryfall/',
              'feat_vec_size': self.feat_vec_size}

        num_train = len(self.partition['train_ids']) if num_train is None else num_train
        num_val = len(self.partition['val_ids']) if num_val is None else num_val

        self.training_generator = train_generator.DataGenerator(self.partition['train_ids'][:num_train], self.file_names, self.seq, **self.generator_params)
        self.validation_generator = train_generator.DataGenerator(self.partition['val_ids'][:num_val], self.file_names, self.seq, **self.generator_params)

    def compile(self, learning_rate=0.001):
        self.opt = tf.keras.optimizers.Adam(learning_rate)
        self.model.compile(optimizer=self.opt, loss=self.loss)

    def set_layer(self, layer_name, is_trainable=True):
        self.model.layers[3].get_layer(layer_name).trainable = is_trainable

    def train(self, epochs, weight_name):
        for i in np.arange(0,epochs):
            H = self.model.fit(x = self.training_generator,validation_data=self.validation_generator,epochs=1)
            self.model.save_weights(card_recognition_model._weights_path+weight_name+'.h5',overwrite=True)

    def load_weights(self, weight_file):
        self.model.load_weights(card_recognition_model._weights_path+weight_file+'.h5',by_name=True,skip_mismatch=True)

    def save_weights(self, weight_file):
        self.model.save_weights(card_recognition_model._weights_path+weight_file+".h5",overwrite=False)

    def compute_feat_vecs(self, weight_file=None):
        if weight_file is not None:
            self.load_weights_pred(weight_file)

        feat_vecs = np.empty((len(self.file_names),self.feat_vec_size),dtype=np.float32)
        feat_dict = {}#{filename: index_feat_vecs}

        self.predict_batch_size = 32
        list_IDs = np.arange(1,len(self.file_names)+1)
        num_batches = int(len(list_IDs) / self.predict_batch_size)

        for i in np.arange(0,num_batches*self.predict_batch_size,self.predict_batch_size):
            images = np.empty((self.predict_batch_size,*self.dim,3),dtype='uint8')
            image_names = np.empty(self.predict_batch_size,dtype=object)
            for j,id in enumerate(list_IDs[i:i+self.predict_batch_size]):
                img = cv.imread(card_recognition_model._train_images +'scryfall/'+ self.file_names[id])
                img = cv.resize(img,(self.dim[1],self.dim[0]))
                images[j] = img
                image_names[j] = self.file_names[id]
            predictions = self.model_pred.predict_on_batch(images)

            for j in np.arange(0,self.predict_batch_size):
                feat_vecs[i+j] = predictions[j]
                feat_dict[image_names[j]] = i + j

            print(i,end='\r')

        from_index = num_batches * self.predict_batch_size
        for i, id in enumerate(list_IDs[from_index:len(self.file_names)]):
            img = cv.imread(card_recognition_model._train_images +'scryfall/'+ self.file_names[id])
            img = cv.resize(img,(self.dim[1],self.dim[0]))
            predict = self.model_pred.predict(img.reshape(1,*self.dim,3))
            feat_vecs[from_index+i]  = predict
            feat_dict[self.file_names[id]] = from_index+i

        return [feat_vecs,feat_dict]

    def predict_image(self, img):
        img = cv.resize(img,(self.dim[1],self.dim[0]))

        predicted_feat_vec = self.model_pred.predict(img.reshape(1,*self.dim,3))[0]
        #predicted_feat_vec = self.augment_test(img)

        distances = predicted_feat_vec - self.feat_vecs
        np.square(distances,out=distances)
        distances = np.add.reduce(distances,axis=1,dtype='float32')

        dist_sort_ind = np.argsort(distances)

        min_distance = distances[dist_sort_ind[0]]
        min_url = self.feat_vecs_urls[dist_sort_ind[0]]
        card_name = self.cardnames[min_url]

        pred_img = cv.imread(card_recognition_model._train_images +'scryfall/'+ min_url)

        return pred_img

    def load_pred_arrays(self, weight_file):
        self.load_weights(weight_file)

        with open(card_recognition_model._train_images+'default-cards-20210421090304.json',encoding="utf8") as myfile:
            data=myfile.read()
        obj = json.loads(data)

        self.cardnames = {}
        for card in obj:
            self.cardnames[card['id']+'.jpg'] = card['name']

        self.feat_vecs = np.load(card_recognition_model._feature_files+weight_file+'/feat_vecs.npy',allow_pickle=True)
        self.feat_dict = np.load(card_recognition_model._feature_files+weight_file+'/feat_dict.npy',allow_pickle=True)
        self.feat_dict = self.feat_dict[()]

        self.feat_vecs_urls = np.empty(len(self.feat_vecs),dtype=object)
        for key,value in self.feat_dict.items():
            self.feat_vecs_urls[value] = key

    def test_performance(self):
        with open(card_recognition_model._test_images+'card_labels.json',encoding="utf8") as myfile:
            data=myfile.read()
        obj = json.loads(data)

        num_matched = 0;
        num_matched_exact = 0;

        for test_id,test_card in enumerate(obj):
            time_start = time.clock()

            filename_test = test_card['file_name']
            cardname_test = test_card['name']
            cardid_test = test_card['id']

            img = cv.imread(card_recognition_model._test_images +'images/'+ filename_test)
            img = cv.resize(img,(self.dim[1],self.dim[0]))

            time_elapsed = (time.clock() - time_start)
            #print("img reading time", time_elapsed, sep =":")

            time_start = time.clock()
            predicted_feat_vec = self.model_pred.predict(img.reshape(1,*self.dim,3))[0]
            #predicted_feat_vec = self.augment_test(img)
            time_elapsed = (time.clock() - time_start)
            #print("feature comp time", time_elapsed, sep =":")


            time_start = time.clock()
            distances = predicted_feat_vec - self.feat_vecs
            np.square(distances,out=distances)
            distances = np.add.reduce(distances,axis=1,dtype='float32')

            time_elapsed = (time.clock() - time_start)
            #print("compute distances", time_elapsed, sep =":")

            time_start = time.clock()
            dist_sort_ind = np.argsort(distances)
            time_elapsed = (time.clock() - time_start)
            #print("sorted distances", time_elapsed, sep =":")


            min_distance=distances[dist_sort_ind[0]]
            min_url = self.feat_vecs_urls[dist_sort_ind[0]]
            card_name = self.cardnames[min_url]

            if cardname_test == card_name:
                num_matched = num_matched+1

            if cardid_test == min_url.partition(".")[0]:
                num_matched_exact = num_matched_exact+1

            print("%2d out of %2d" % (test_id,len(obj)),end='\r')

        print("Accuracy card name recognition: ",num_matched/len(obj))
        print("Accuracy exact id recognition: ",num_matched_exact/len(obj))


    def augment_test(self, img):
        test_batch_size = 32

        temp_seq = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0.0, 1.0)),
            iaa.GammaContrast((0.2, 2.5)),
            iaa.Affine(scale=(0.95,1.10))#old was -15,35
        ])

        img_batch = np.tile(img,(test_batch_size,1,1,1))
        img_augmented = temp_seq(images=img_batch)

        predicted_feat_vecs = self.model_pred.predict(img_augmented)

        avg_feat_vec = np.median(predicted_feat_vecs,axis=0)

        return avg_feat_vec
