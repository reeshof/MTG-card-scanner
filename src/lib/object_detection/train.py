import utils.util as util

import json
import numpy as np
import cv2 as cv

def addBoxToHeatMap(heatmap, centerPoint, gaussian_width, angle):
    heatCenterPoint = np.zeros((heatmap.shape[0],heatmap.shape[1],1),np.float32)
    heatCenterPoint.itemset((centerPoint[0],centerPoint[1],0),1)
    #add keypoint to image
    #the width and height of gaussian should be determined by the width and height of boundingbox
	#for the time being this is fixed to square of size (55,55)
	#sigma in x and y direction is fixed to a third of width, height (as is done in the centerpoint paper)
    gaussian_width = util.round_up_to_odd(gaussian_width).astype(int)
    heatCenterPoint = cv.GaussianBlur(heatCenterPoint,(gaussian_width,gaussian_width),gaussian_width/3.0,gaussian_width/3.0)
    #opencv is using a normalized gaussian. paper uses an unnormlized gaussian.
	#thus the location of the keypoint should have a value of 1.
    scalefactor = 1 / heatCenterPoint.item(centerPoint[0],centerPoint[1])
    heatCenterPoint *= scalefactor

    rot_mat = cv.getRotationMatrix2D((centerPoint[1],centerPoint[0]), angle, 1.0)
    heatCenterPoint = cv.warpAffine(heatCenterPoint, rot_mat, heatCenterPoint.shape[1::-1], flags=cv.INTER_LINEAR)

    #set keypoint to 1 to be sure, in case of possible rounding errors
	#loss function relies on this keypoint location to be exactly 1
    heatCenterPoint.itemset((centerPoint[0],centerPoint[1]),1)
    heatCenterPoint = heatCenterPoint.reshape(heatCenterPoint.shape[0],heatCenterPoint.shape[1],1)
    #not much information lost by multiplying with the scalefactor.
	#I.e., the difference between applying the gaussian and then multiplying, and applying
	#the gaussian when the center value is this scale factor.

    heatmap = cv.max(heatmap,heatCenterPoint)#collect these splattered regions in one image)

    return heatmap

def getHeatMap(img, centerPoints):
	heat = np.zeros((img.shape[0],img.shape[1],1),np.float32)

	for centerPoint in centerPoints:
		heat = addBoxToHeatMap(heat,centerPoint)

	return heat

def readJSON(trainingpath):
    with open(trainingpath + 'instances_default.json') as myfile:
        data=myfile.read()
    obj = json.loads(data)
    return obj

def get_dictionaries(train_split,datasetPath):
    obj = readJSON(datasetPath)

    n_images = obj['images']


    annotations = {}
    file_names = {}

    for image in obj['images']:
        file_names[image['id']] = image['file_name']
        annotations[image['id']] = []

    for annotation in obj['annotations']:
        segment = annotation['segmentation'][0]
        if len(segment) > 8:
            segment = segment[0:8]
        annotations[annotation['image_id']].append(segment)

    partition = {'train_ids': [], 'val_ids': []}

    indices = np.arange(1,len(obj['images'])+1)
    util.shuffleArray(indices,0)

    n_train_images = int(train_split*len(indices))
    partition['train_ids'] = indices[0:n_train_images]
    partition['val_ids'] = indices[n_train_images:len(indices)+1]

    return (file_names,partition,annotations)

#getTrainTest is not used for the object detection model, instead the train generators are used for training
def getTrainTest(shape, datasetPath= "data/obj_det_training/"):
#shape: the image resolution in which the imported dataset should be converted (height,width,depth)
    obj = readJSON(datasetPath)

    #fixed size resolution can be changed and simply resize every input to the required size
    images = np.empty((len(obj['images']),shape[0],shape[1],3),dtype='uint8')
    heatmaps = np.empty((len(obj['images']),shape[0],shape[1],1),dtype=np.float32)
    anglemaps = np.empty((len(obj['images']),shape[0],shape[1],1),dtype=np.float32)
    whmaps = np.empty((len(obj['images']),shape[0],shape[1],2),dtype=np.float32)

    centerpoints = np.empty((len(obj['images']),3),dtype=np.float32)

    segmentMasks = np.empty((len(obj['images']),8),dtype=float)#holds the segmentation mask

    for image in obj['images']:
        heatmaps[image['id']-1] = np.zeros((shape[0],shape[1],1),np.float32)
        img = cv.imread(datasetPath +'train_set/' + image['file_name'])
        img = cv.resize(img,(shape[1],shape[0]))
        images[image['id']-1] = img

        resfactor = [0,0]
        resfactor[0] = shape[0] / image['height']
        resfactor[1] = shape[1] / image['width']
        segmentMasks[image['id']-1] = np.tile(np.array(resfactor),4)

    for annotation in obj['annotations']:
        segment = annotation['segmentation'][0]
        if len(segment) > 8:
            segment = segment[0:8]
        segment = np.multiply(segment,segmentMasks[annotation['image_id']-1])
        centerpoint = np.flip(np.round(util.computerCenterPointPolygon(segment),0).astype(int))

        centerpoints[annotation['image_id']-1] = [centerpoint[0],centerpoint[1],0]

        object_size = util.polygonbox_width_height(segment)
        gaussian_width = util.gaussian_radius(object_size)

        whmaps[annotation['image_id']-1].itemset((centerpoint[0],centerpoint[1],0), object_size[0])
        whmaps[annotation['image_id']-1].itemset((centerpoint[0],centerpoint[1],1), object_size[1])

        angle = util.polygonbox_angle(segment)

        anglemaps[annotation['image_id']-1].itemset((centerpoint[0],centerpoint[1],0), angle +0.00000001)
        heatmaps[annotation['image_id']-1] = addBoxToHeatMap(heatmaps[annotation['image_id']-1], centerpoint, gaussian_width,angle.astype(int)).reshape(shape[0],shape[1],1)

    return (images,heatmaps,anglemaps,whmaps)
