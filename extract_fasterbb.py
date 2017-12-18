import numpy as np
from PIL import Image
import caffe,cv2,pickle,os
#http://adilmoujahid.com/posts/2016/06/introduction-deep-learning-python-caffe/
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
def transform_img(img, img_width, img_height):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img

GPU_ID = 1 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)

"""
#CNN
net = caffe.Net('/auto/master04/jacky82226/cnn/caffe/models/bvlc_alexnet/deploy.prototxt',
                '/auto/master04/jacky82226/cnn/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel',
                caffe.TEST)

layer_list=["pool5","fc6","fc7"]
output_root = "visual_feature/CNN"

#layer_list=["conv5","conv4","conv3"]
"""

# placeCNN

net = caffe.Net('/tmp3/jacky82226/placeCNN/deploy_alexnet_places365.prototxt',
                '/tmp3/jacky82226/placeCNN/alexnet_places365.caffemodel',
                caffe.TEST)

output_root = "visual_feature/placeCNN"
layer_list=["pool5", "fc6","fc7", "fc8", "prob"]
#layer_list=["fc6"]

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)


transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

frame_dir="frame/all/"
proposal_list = ["faster_bb"]
proposal_max = 200

for img_name in os.listdir(frame_dir):
	img_name=img_name.replace(".jpg","").replace(".png","")
	origin_img = cv2.imread(os.path.join(frame_dir, img_name+'.jpg'), cv2.IMREAD_COLOR)

	for proposal_dir in proposal_list:
		proposal_num = 0
		with open(os.path.join(proposal_dir, img_name+".txt"), 'r') as ff:
			output_dir = os.path.join(output_root, proposal_dir)
			for linee in ff:
				token=linee.strip().split()
				bb=[int(float(token[0])), int(float(token[1])), int(float(token[2])), int(float(token[3]))]
				box_score=float(token[4])
				[bb_width,bb_height]=[bb[3]-bb[1],bb[2]-bb[0]]

				img=origin_img[bb[1]:bb[3],bb[0]:bb[2]]

				print(bb)
				img = transform_img(img,227,227)

				net.blobs['data'].data[...] = transformer.preprocess('data', img)

				out = net.forward()
				#pred_probas = out['prob']

				for layer in layer_list:
					output_layer = os.path.join(output_dir, layer)

					if not os.path.exists(output_layer):
					    os.makedirs(output_layer)
					with open(os.path.join(output_layer, img_name+"_"+str(bb[0])+"_"+str(bb[1])+"_"+str(bb[2])+"_"+str(bb[3])+".pkl"), 'wb') as ff:
						pickle.dump(net.blobs[layer].data,ff)
				proposal_num += 1
				if proposal_num >= proposal_max:
					break
				#print pred_probas.argmax()
				#out = net.blobs['score'].data[0].argmax(axis=0)

				# features = net.blobs['fc6'].data.copy()

				#l2 norm distance alexnet pretrained-model
				#pool5
