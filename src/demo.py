import _init_paths
import card_recognition.card_recognition as rec
import object_detection.object_detection as obj
import utils.util as util

import time
import cv2 as cv
import numpy as np

def recognition_model(feat_vec_size1):
    model = rec.card_recognition_model(feat_vec_size=feat_vec_size1)

    layers_top = ['block7a_project_conv', 'top_conv', 'block7a_expand_conv', 'block6d_project_conv', 'block6c_project_conv', 'block6b_project_conv', 'block6a_project_conv', 'block6b_expand_conv']
    layers_mid = ['block6a_expand_conv','block5c_project_conv','block5c_expand_conv','block5b_dwconv','block5b_expand_conv','block5a_project_conv']
    layers_bot = ['block5a_expand_conv','block4c_project_conv','block4c_expand_conv','block4b_project_conv','block4b_expand_conv','block4a_project_conv','block4a_expand_conv']

    for layer in layers_top:
        model.model.layers[3].get_layer(layer).trainable = True
    for layer in layers_mid:
        model.model.layers[3].get_layer(layer).trainable = True
    #for layer in layers_bot:
        #model.model.layers[3].get_layer(layer).trainable = True
    #
    model.compile(0.00001)

    return model

def detection_model(dim=(512,384)):
    model = obj.object_detection_model(dim)
    model.load_weights('obj_detection_2')

    return model

def demo_detection_test(filename ='IMG_0144.png'):
    model = obj.object_detection_model((512,384))
    model.load_weights('obj_detection_2')

    img = cv.imread(filename)

    #returns the resized image (that corresponds with model dimensions), and the prediction maps
    [resized_img,pred] = model.get_predictions(img)

    #From the prediction maps, obtain the bounding boxes with orientation
    bbs = obj.get_boundingboxes(pred,(512,384))

    util.draw_bbs(resized_img,bbs)
    util.show_img(resized_img)

def demo_detection_train():
    model = obj.object_detection_model((512,384))

    layers_top = ['block7a_project_conv', 'top_conv', 'block7a_expand_conv', 'block6d_project_conv', 'block6c_project_conv', 'block6b_project_conv', 'block6a_project_conv', 'block6b_expand_conv']
    layers_mid = ['block6a_expand_conv','block5c_project_conv','block5c_expand_conv','block5b_dwconv','block5b_expand_conv','block5a_project_conv']
    layers_bot = ['block5a_expand_conv','block4c_project_conv','block4c_expand_conv','block4b_project_conv','block4b_expand_conv','block4a_project_conv','block4a_expand_conv']

    for layer in layers_top:
        model.set_layer(layer,True)
    for layer in layers_mid:
        model.set_layer(layer,True)

    model.load_generators()#loads training and validation data
    model.compile(0.001)
    model.load_weights('obj_detection_2')#load weights to start from a previous trained model

    model.train(5,"demo_detection_1")

def demo_recognition_test():
    model = recognition_model(256)
    model.load_pred_arrays('img_recognition_4_1')
    model.test_performance()

def demo_recognition_test_single(model,img):
    pred_image = model.predict_image(img)
    util.showimg(pred_image)

def demo_recognition_train():
    model = recognition_model(256)
    model.load_weights("img_recognition_3_4_5")

    model.load_generators(batch_size=4)#low batch size because of memory issues
    model.train(5,'temp_weights')

def initialize(dim):
    model_det = detection_model(dim)
    model_rec = recognition_model(256)
    model_rec.load_pred_arrays('img_recognition_4_1')
    cam = cv.VideoCapture(0)

    return (model_det,model_rec,cam)

def cam_loop(model_det,model_rec,cam, dim):
    while True:
        time_start = time.clock()

        ret_val, img = cam.read()

        [i,pred] = model_det.get_predictions(img)
        bbs = obj.get_boundingboxes(pred,dim)
        drawi = i.copy()

        util.draw_bbs(drawi,bbs)

        regions = model_det.get_regions(i,pred)

        for region in regions:
            if region.shape[0] > 10 and region.shape[1] > 10:
                pred_image = model_rec.predict_image(region)
                cv.imshow("region",region)
                cv.imshow("pred_img",pred_image)

        cv.imshow("myimg",drawi)

        time_elapsed = (time.clock() - time_start)
        print("Frame processing time: ",time_elapsed)

        if cv.waitKey(1) == 27:
            break  # esc to quit

    cv.destroyAllWindows()
