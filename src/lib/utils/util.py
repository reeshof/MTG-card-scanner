import numpy as np
import matplotlib.pyplot as plt
import math
import imgaug as ia
import cv2 as cv

def round_up_to_odd(f):
    return np.ceil(f) // 2 * 2 + 1

def computeCenterPoint(boundingBox):
#boundingBox = [topLeftx, toplefty, width, height]
	xAverage = boundingBox[0] + boundingBox[0]+boundingBox[2]
	yAverage = boundingBox[1] + boundingBox[1]+boundingBox[3]
	return [xAverage/2,yAverage/2]#return the centerpoint

def computerCenterPointPolygon(polygon):
#polygon [Vertex1_x, vertex1_y, .. , vertex4_x, vertex4_y]
	xtotal = polygon[0]+polygon[2]+polygon[4]+polygon[6]
	ytotal = polygon[1]+polygon[3]+polygon[5]+polygon[7]
	return [xtotal/4,ytotal/4]#return the centerpoint

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def clockwise_angle(v):
#angle between the vector [0,-1] and v, the total clockwise angle
#if x direction is positive then its simply the angle, if its negative its the calculated angle + p
	angle = angle_between([0,-1],v)
	if(v[0] < 0):
		angle = np.pi + (np.pi-angle)
	return angle

def angle_distance(a1,a2):
	dist = np.absolute(a1 - a2)
	dist2 = np.absolute(a1+2*np.pi - a2)
	dist = np.minimum(dist,dist2)

	return dist

def polygonbox_angle(box):
#given a bounding box defines by its 4 corners [x1,y1,..,x4,y4] return its angle with the [0,-1] vector
	dir_vec= np.subtract([box[2],box[3]],[box[4],box[5]])
	return clockwise_angle(dir_vec)

def polygonbox_width_height(box):
#given a bounding box defines by its 4 corners [x1,y1,..,x4,y4] return its width and height
	p1 = [box[0],box[1]]
	p2 = [box[2],box[3]]
	p3 = [box[4],box[5]]
	p4 = [box[6],box[7]]

	w1 = np.linalg.norm(np.subtract(p2,p1),ord=None)
	w2 = np.linalg.norm(np.subtract(p4,p3),ord=None)

	h1 = np.linalg.norm(np.subtract(p3,p2),ord=None)
	h2 = np.linalg.norm(np.subtract(p4,p1),ord=None)

	return ((w1+w2)/2,(h1+h2)/2)


def rotate(origin, point, angle):
    """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def shuffleArray(data, seed):
    rng = np.random.default_rng(seed)
    rng.shuffle(data)

def shuffleData(images,heatmaps,anglemaps,whmaps,seed):
    """
        Shuffle the input data using the same seed such that each of the input is sorted the same
    """
    shuffleArray(images, seed)
    shuffleArray(heatmaps, seed)
    shuffleArray(anglemaps, seed)
    shuffleArray(whmaps, seed)

def plotData(loss,val_loss,title,name,model_history_path,ylim=None):
    plt.plot(loss)
    plt.plot(val_loss)
    #plt.yscale("log")
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if ylim is not None:
        plt.ylim(ylim)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(model_history_path+name+'/'+title)
    plt.close()

def minmaxLoss(history):
    """
        Prints the mininum value for each loss in a keras model training history
    """
    for key in history.history.keys():
        print(key + ': ' + str(np.min(history.history[key])))

def gaussian_radius(det_size, min_overlap=0.7):
    """
        Returns the width a gaussian kernel should have for a given bounding box size.
            https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    """
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2

    return min(r1, r2, r3)

def bbs_to_polygons(boundingboxes):
	polygons = []

	for boundingbox in  boundingboxes:
		bba = []
		for i in range(0,len(boundingbox),2):
			x = boundingbox[i]
			y = boundingbox[i+1]
			bba.append((x,y))
		poly = ia.Polygon(bba)
		polygons.append(poly)

	return polygons

def polygons_to_bbs(polygons):
	boundingboxes = []
	for polygon in polygons:
		bb = [y for x in polygon for y in x]
		boundingboxes.append(bb)
	return boundingboxes

def point_outside_bounds(point,image_dim):
    if point[0] >= image_dim[0] or point[0] < 0:
        return True
    if point[1] >= image_dim[1] or point[1] < 0:
        return True
    return False

def set_kernel_center(kernel,value):
    x = int(kernel.shape[0]/2)
    y = int(kernel.shape[1]/2)
    kernel[x][y] = value
    return kernel

def find_local_maximum(heatmap,kernel_size=(31,31)):
    dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT,kernel_size)
    dilate_kernel = set_kernel_center(dilate_kernel,0)

    dilated_heatmap = cv.dilate(heatmap,dilate_kernel)
    local_max_image = heatmap > dilated_heatmap

    local_max_indices = np.argwhere(local_max_image)
    local_max_values = [heatmap[x][y] for (x,y) in local_max_indices]

    local_max_zipped = sorted(zip(local_max_values,local_max_indices), key=lambda x: x[0],reverse=True)

    return local_max_zipped

def compute_orientation_vec(angle_sin):
    """
        Returns the directional vector given the sin of the angle.
        (the cos of the angle is assumed to be negative, i.e. the vector is pointing upwards in the image)
    """
    angle_cos = np.sqrt(1-angle_sin*angle_sin)
    return [-angle_cos,angle_sin]#this is a vector of unit length

def extract_roi(img,centerpoint,width,height,angle,output_size=(204,146)):
    """
        Given an image and a non axis aligned bounding box, return the image region corresponding to this bounding
        box cropped out of the original image and resize to the desired output_size
    """
    rot_mat = cv.getRotationMatrix2D((centerpoint[1],centerpoint[0]), np.degrees(angle), 1.0)
    img = cv.warpAffine(img, rot_mat, (img.shape[1],img.shape[0]), flags=cv.INTER_LINEAR)

    topleftpoint = [centerpoint[0]- (height / 2.0), centerpoint[1] - (width/2.0)]
    topleftpoint = [int(topleftpoint[0]),int(topleftpoint[1])]

    cropped_image = img[topleftpoint[0]:topleftpoint[0]+int(height),topleftpoint[1]:topleftpoint[1]+int(width)]

    return cropped_image

def show_img(img):
    cv.imshow("myimg",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def draw_bbs(img,bbs):
    for bb in bbs:
        botright = np.array(bb[1]).astype(int)
        topleft = np.array(bb[0]).astype(int)
        cv.rectangle(img,(topleft[0],topleft[1]),(botright[0],botright[1]),(0,255,0),2)

        og = np.array(bb[2]).astype(int)
        dest = np.array(bb[3]).astype(int)
        cv.line(img,(og[1],og[0]),(dest[1],dest[0]),(255,0,0),2)
