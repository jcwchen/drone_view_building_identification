import cv2, random
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import IPython
import multiprocessing as mp

random.seed(1223)

class TripletData:
    def __init__(self, anchor, positive, negative):
        # load origin images
        self.anchor   = anchor
        self.positive = positive
        self.negative = negative

class TripletCrossViewData:
    def __init__(self, anchor, positive, negative):
        # load origin images
        self.anchor = {}
        
        for query_dir in anchor:
            self.anchor[query_dir] = open_img(anchor[query_dir])

        self.positive = open_img(positive)
        self.negative = open_img(negative)


class PairData:
    def __init__(self, pair1, pair2, label):
        # load origin images
        self.pair1 = transform_img(pair1)
        self.pair2 = transform_img(pair2)
        self.label = label

class Dataset:
    def __init__(self, data):
        self.data = []
        for d in data:
            self.data.append(d)
        self.data_length = len(self.data)
        self.reset_sample()

    def sample(self, batch_size):
        data_list = []
        isnextepoch = False 
        if self.sample_top+batch_size < self.data_length:
            self.sample_top += batch_size
            data_list = self.data[self.sample_top-batch_size:self.sample_top]
        else:
            isnextepoch = True
            data_list = self.data[self.sample_top:self.data_length]
            self.sample_top = self.data_length
        x = []
        y = []
        z = []
        for d in data_list: 
            x.append(d.anchor)
            y.append(d.positive)
            z.append(d.negative)
        return np.array(x), np.array(y), np.array(z), isnextepoch, self.sample_top - len(data_list), self.sample_top


    def sample_path2img(self, batch_size, iscross=False):
        data_list = []
        isnextepoch = False 
        if self.sample_top+batch_size < self.data_length:
            self.sample_top += batch_size
            data_list = self.data[self.sample_top-batch_size:self.sample_top]
        else:
            isnextepoch = True
            data_list = self.data[self.sample_top:self.data_length]
            self.sample_top = self.data_length
        x_search = []
        x_street = []
        x_aerial = [] 
        y = [] 
        z = []

        def load(data_list, q):                    
            for d in data_list:
                q.put(open_triplet(d))
            # finishing
            q.put(None)
            q.close()
        def open_triplet(d):
            try:
                if not iscross:
                    return TripletData(open_img(d.anchor),  open_img(d.positive), open_img(d.negative))
                else:
                    return TripletCrossViewData(d.anchor, d.positive, d.negative)
            except:
                print(d.anchor['search']+"\n"+d.positive+"\n"+d.negative)
        # > 2* batch_size
        queue_size = 4 * batch_size
        q = mp.Queue(maxsize=queue_size)

        # background loading Shapes process
        p = mp.Process(target=load, args=(data_list, q))
        # daemon child is killed when parent exits
        p.daemon = True
        p.start()


        for i in xrange(batch_size):
            
            # print 'q size', q.qsize() 

            s = q.get()

            # queue is done
            if s == None: 
                break
            if not iscross:
                x_search.append(s.anchor)
            else:
                x_search.append(s.anchor['search'])
                x_street.append(s.anchor['streetview_clean'])
                x_aerial.append(s.anchor['aerial_clean'])
            y.append(s.positive)
            z.append(s.negative)
        if not iscross:
            return np.array(x_search), np.array(y), np.array(z), isnextepoch, self.sample_top - len(data_list), self.sample_top
        else:
            return {'search': np.array(x_search), 'streetview_clean': np.array(x_street), 'aerial_clean': np.array(x_aerial)
            }, np.array(y), np.array(z), isnextepoch, self.sample_top - len(data_list), self.sample_top

    def reset_sample(self):
        self.sample_top = 0
        self.data_length = len(self.data)
        random.shuffle(self.data)

    def remove(self, start, end):
        del self.data[start:end]    
        self.sample_top = start
        self.data_length = len(self.data)
        print("remove invalid triplet... data remain {}".format(len(self.data)))
        
        if len(self.data) <=0:
            print("Terminate with no more data")
            return 1 # terminate
        elif self.sample_top == self.data_length:
            print("Cut tail")
            return -1

        return 0

def open_img(path):
    return transform_img(cv2.imread(path, cv2.IMREAD_COLOR))

def transform_img(img, img_width = 227 , img_height= 227):

    #Histogram Equalization
    """
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    """

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

def proposal_enlarge(img, bb, ratio):

    height, width, rgb = img.shape

    x_center = (bb[2] + bb[0])/2
    y_center = (bb[3] + bb[1])/2

    bb_width_half  = (bb[2] - bb[0]) * ratio/2
    bb_height_half = (bb[3] - bb[1]) * ratio/2

    if x_center - bb_width_half < 0:
        x1 = 0
    else:
        x1 = x_center - bb_width_half
    if x_center + bb_width_half > width:
        x2 = width
    else:
        x2 = x_center + bb_width_half
    if y_center - bb_height_half < 0:
        y1 = 0
    else:
        y1 = y_center - bb_height_half
    if y_center + bb_height_half > height:
        y2 = height
    else:
        y2 = y_center + bb_height_half
    return [int(x1), int(y1), int(x2), int(y2)]

def img_augmentation(img, sample_num, save_dir):
    
    datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.5, #0.2
        height_shift_range=0.5, #0.2
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='reflect')


    img_batch = []
    i = 0
    img = img.reshape((1,) + img.shape)

    for batch in datagen.flow(img, 
                              batch_size=1,
                              save_to_dir=save_dir,  
                              save_prefix='test', 
                              save_format='jpg'):
        i += 1
        img_batch.append(batch[0])
        if i >= sample_num:
            break

    return img_batch
