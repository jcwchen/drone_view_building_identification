import os, sys, math, operator, cv2, glob
import numpy as np
from math import radians, cos, sin, asin, sqrt, fabs
from scipy.spatial.distance import cdist
from scipy import spatial

import IPython
try:
    from PIL import Image
except:
    import Image
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def tolist(self):
        return [self.x, self.y]

# gps ditance
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two ps 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

# calculate angle

def rotation_matrix(v, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """

    t=math.radians(theta)

    return [v[0]*math.cos(t)+v[1]*math.sin(t), -v[0]*math.sin(t)+v[1]*math.cos(t)]


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return sqrt(dotproduct(v, v))

def yawtoangle(y):
	if y<0:
		return 360+y
	else:
		return y


def angle(v1, v2):
  a=math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
  if(np.dot(v1[0], v2[1])-np.dot(v1[1], v2[0])<0):
  	a=-a
  return a

h_max, s_max, v_max=(18, 4, 3) #HSV
r_max, g_max, b_max=(4, 4, 4) #RGB 64bins

# hsv color feature
def rgb2hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    if(h>=360):
    	h=359.9999
    if(s>=1):
    	s=0.99999
    if(v>=1):
    	v=0.99999
    return int(h/20), int(s*4), int(v*3)
    #(360, 1, 1) to (18, 4, 3)

def get_hsv_histo(img):

	histo = [[[0 for z in range(v_max)] for y in range(s_max)] for x in range(h_max)]
	width, height = img.size
	img = img.convert('RGB')

	for x in range(width):
		for y in range(height):
			R, G, B=img.getpixel((x, y))
			H, S, V=rgb2hsv(R, G, B)
			histo[H][S][V]+=1/float(width*height) #normalize for different size 
	return histo

def get_rgb_histo(filename):

	histo = [[[0 for z in range(b_max)] for y in range(g_max)] for x in range(r_max)]
	img=Image.open(filename)
	width, height = img.size
	img = img.convert('RGB')

	for x in range(width):
		for y in range(height):
			R, G, B=img.getpixel((x, y))
			histo[R/64][G/64][B/64]+=1/float(width*height) #normalize for different size 
	return histo

def l1_distance(method, query, data):
	difference=0
	if(method==1): #HSV dimension
		for x in range(h_max):
			for y in range(s_max):
				for z in range(v_max):
					difference+=math.fabs(query[x][y][z]-data[x][y][z])
	elif(method==2): #RGB dimension for 64bins
		for x in range(len(query)):
			difference+=math.fabs(query[x]-data[x])
	return difference


def color_dis(query_histo, data_histo):
	return l1_distance(1, query_histo, data_histo)

 
def likelyscale(bb, width, height):
    # distance to bottm affect the width of window
    return (math.fabs(float(bb[3]/height)**2*width-(bb[2]-bb[0])))/width

def likelydistance(gps, center, bb, height, width, x_dis, isx,isprint=False):
    # GPS distance affect distance to middle bottm
    """
    gg=haversine(gps[0], gps[1], center[0], center[1])
    pic_dis=sqrt(((bb[0]+bb[2])/2-width/2)**2+(bb[3]-height)**2)/max_dis
    print(str(gg)+" "+str(pic_dis))
    return math.fabs(gg-pic_dis)"""
    #map_dis = haversine(gps[1], gps[0], center[1], center[0])/1.5
    #drone_dis = sqrt(((bb[0]+bb[2])/2-width/2)**2+(bb[3]-height)**2)/max_dis
    if isx:
        map_x_dis = math.fabs(x_dis)*310
        drone_x_dis = math.fabs((float(bb[0]+bb[2])/2-width/2)/width)
        #if isprint:
            #print("{}, {}, {}".format(map_x_dis, drone_x_dis, map_x_dis-drone_x_dis))
        return math.fabs(map_x_dis-drone_x_dis) #0.28
    else:
        map_dis = haversine(gps[1], gps[0], center[1], center[0])
        map_y_dis = sqrt(map_dis**2-x_dis**2)
        drone_y_dis = math.fabs(float(height - bb[3])/height)

        #if isprint:
            #print("{}, {}, {}".format(map_y_dis, drone_y_dis, map_y_dis-drone_y_dis))
        return math.fabs(map_y_dis-drone_y_dis) #0.67


def simple_iou(x, y):
    #SI = Max(0, Max(XA2, XB2) - Min(XA1, XB1)) * Max(0, Max(YA2, YB2) - Min(YA1, YB1))
    intersect=max(0, min(x[2], y[2])-max(x[0], y[0]))*max(0, min(x[3], y[3])-max(x[1], y[1]))
    return float(intersect/((x[2]-x[0])*(x[3]-x[1])))


def iou(x, y):
    #SI = Max(0, Max(XA2, XB2) - Min(XA1, XB1)) * Max(0, Max(YA2, YB2) - Min(YA1, YB1))
    intersect=max(0, min(x[2], y[2])-max(x[0], y[0]))*max(0, min(x[3], y[3])-max(x[1], y[1]))
    return float(intersect)/((x[2]-x[0])*(x[3]-x[1])+(y[2]-y[0])*(y[3]-y[1])-intersect)

def eval_angle(width, height, x, y):
    origin=[width/2, height]
    return math.fabs(math.degrees(angle([(x[0]+x[2])/2-origin[0], x[3]-origin[1]], [(y[0]+y[2])/2-origin[0], y[3]-origin[1]])))

def likelyvisual(target, proposal):
    sigma=0
    if len(target.shape)==4: # pool5
        for dim_0 in range(target.shape[0]):
            for dim_1 in range(target.shape[1]):
                for dim_2 in range(target.shape[2]):
                    for dim_3 in range(target.shape[3]):
                        sigma+=l2_distance(target[dim_0][dim_1][dim_2][dim_3], proposal[dim_0][dim_1][dim_2][dim_3])
    elif len(target.shape)==2: # fc6 and fc7
        for dim_0 in range(target.shape[0]):
            for dim_1 in range(target.shape[1]):
                sigma+=l2_distance(target[dim_0][dim_1], proposal[dim_0][dim_1])
    return sigma

def likelyvisual_multiple(target, proposal):
    sigma=0
    for streetview_num in range(target.shape[0]):
        #sigma+=np.linalg.norm(target[streetview_num]-proposal)
        sigma+=np.sqrt(np.sum(np.absolute(target[streetview_num]-proposal)))
    return sigma


def likelyvisual_multiple_min(target, proposal):
    min_dis=sys.maxint
    streetview_num=range(target.shape[0])
    for streetview_num in range(target.shape[0]):
        dis=np.sqrt(np.sum(np.absolute(target[streetview_num]-proposal)))
        if min_dis>dis:
            min_dis=dis
    return min_dis

def l2_distance(x, y):
    return (x-y)**2

def cos_sim(a_list, b_list):
    return 1 - spatial.distance.cosine(a_list, b_list)


def normalization(array):
    norm=[]
    array_max=np.amax(array)
    array_min=np.amin(array)
    div=array_max-array_min
    for a in array:
        norm.append((a-array_min)/div)
    return norm

def dis_to_sim(array):
    #array=normalization(array) #general norm
    array_max=np.amax(array)
    sim=[]
    for a in array:
        sim.append(1-a/array_max)
    #sim=sigmoid_array(sim)
    sim=normalization(sim)
    return sim
    #return sim_to_borda(sim)

def sim_to_borda(array):
    borda=[]
    for i in range(len(array)):
        borda.append([i, array[i]])

    borda.sort(key=lambda x: float(x[1]))
    for score in range(len(borda)):
        borda[score][1]=score
    borda.sort(key=lambda x: float(x[0]))
    borda_score=np.array(borda)
    return borda_score[:, 1]

def sigmoid_array(array):
    sig=[]
    for a in array:
        sig.append(sigmoid(a))
    return sig

def sigmoid(x):
    return float(1/(1+math.exp(-x)))

def points2line(p1, forward):    
    """
    p1, p2 [x, y]
    """
    if forward.x!=0:
        a = forward.y / forward.x
    else:
        a = sys.maxint

    b  = p1.y - a * p1.x

    return [a, b]



def disofpoint2line(p, origin_p, forward):
    """
    p [x, y]
    line [a, b, forward]  y = ax + b, forward vector
    """
    [a, b] = points2line(origin_p, forward)
    sign = np.cross(gpsvecotr(p, origin_p), forward.tolist())
    
    sign = 1 if sign >= 0 else -1
    dis = fabs(a*p.x-p.y+b)/(a*a+1)
    return sign*dis

def gpsvecotr(p1, p2):
    sign = 1 if p1.x >= p2.x else -1
    vec_x = sign*haversine(p1.x, p2.y, p2.x, p2.y)

    sign = 1 if p1.y >= p2.y else -1
    vec_y = sign*haversine(p1.x, p1.y, p1.x, p2.y)

    return [vec_x, vec_y]

def yawtovector(y):
    y = 360 + y if y < 0 else y
    return Point(math.sin(math.radians(y)), math.cos(math.radians(y)))

def DCG(score_list):
    score = score_list[0]
    for s in score_list[1:]:
        score += float(s)/math.log(s, 2)
    return score

