import numpy as np
import os,sys
import tensorflow as tf
import os.path as osp
import multiprocessing as mp

import config
import similarity
import model
import IPython
from data import transform_img
import cv2, pickle



flags = tf.app.flags

flags.DEFINE_string('feature', 'fc6', 'Extract which layer(pool5, fc6, fc7)')
flags.DEFINE_string('model_dir', None, 'Model directory')



flags.DEFINE_integer('batch_size', 128, 'Value of batch size')
flags.DEFINE_integer('p', 200, 'Size of proposals')
FLAGS = flags.FLAGS

#layer_list = ["pool5"]
layer_list = [FLAGS.feature]


output_root = "visual_feature/triplet"
query_dir = ["search", "streetview_clean", "aerial_clean"]
proposal_max = FLAGS.p

#query_dir = ["image_gt"]


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def write_pkl(pkl, sess, pred):
    index = 0
    while index*FLAGS.batch_size<len(pkl):
        img = []
        for p in pkl[index*FLAGS.batch_size:(index+1)*FLAGS.batch_size]:
            img.append(p[0])
        out = sess.run([pred], feed_dict={img_search: img})
        out = np.array(out[0])

        p_i = 0
        for p in pkl[index*FLAGS.batch_size:(index+1)*FLAGS.batch_size]:
            p.append(out[p_i])
            p_i += 1


        for p in pkl[index*FLAGS.batch_size:(index+1)*FLAGS.batch_size]:
            with open(os.path.join(p[1]), 'wb') as ff:
                pickle.dump(p[2], ff)
        
        index += 1
        
    print(len(pkl))

def write_cross_pkl(pkl, sess, cross_pred):
    index = 0
    while index*FLAGS.batch_size<len(pkl):
        search = []
        street = []
        aerial = []
        for p in pkl[index*FLAGS.batch_size:(index+1)*FLAGS.batch_size]:
            search.append(p[0]['search'])
            street.append(p[0]['streetview_clean'])
            aerial.append(p[0]['aerial_clean'])
        try:
            out = sess.run([cross_pred], feed_dict={img_search: search, img_street: street, img_aerial: aerial})
        except:
            IPython.embed()
        out = np.array(out[0])

        p_i = 0
        for p in pkl[index*FLAGS.batch_size:(index+1)*FLAGS.batch_size]:
            p.append(out[p_i])
            p_i += 1


        for p in pkl[index*FLAGS.batch_size:(index+1)*FLAGS.batch_size]:
            with open(os.path.join(p[1]), 'wb') as ff:
                pickle.dump(p[2], ff)
        
        index += 1
        
    print(len(pkl))

with tf.Graph().as_default(), tf.Session(config=config) as sess:
    img_search  = tf.placeholder('float32', shape=(None, 227, 227, 3))
    img_street  = tf.placeholder('float32', shape=(None, 227, 227, 3))
    img_aerial  = tf.placeholder('float32', shape=(None, 227, 227, 3))



    cross_feature = model.inference_crossview([img_search, img_street, img_aerial]
        , 1, FLAGS.feature, False)
    
    norm_cross_pred = model.feature_normalize([cross_feature])
    cross_pred = norm_cross_pred[0]
    
    feature = model.inference(img_search, 1, FLAGS.feature)


    norm_pred = model.feature_normalize([feature])
    pred = norm_pred[0]    

    saver = tf.train.Saver()
    if FLAGS.model_dir:
        saver.restore(sess, FLAGS.model_dir)
    else:
        saver.restore(sess, 'model/{}/model_final'.format(FLAGS.feature))

    pkl_list = {}

    if True:
        output_dir = os.path.join(output_root, "cross")

        for img_name in os.listdir(query_dir[0]):
            if img_name.find('.jpg')==-1: #is a directory
                continue

            img_name=img_name.replace(".jpg","").replace(".png","")
            print(img_name)
            img = {}
            for query in query_dir:
                img[query] = cv2.imread(os.path.join(query, img_name+'.jpg'), cv2.IMREAD_COLOR)
                img[query] = transform_img(img[query], 227,227)


            for layer in layer_list:
                output_layer = os.path.join(output_dir, layer)
                if not layer in pkl_list:
                    pkl_list[layer] = []
                if not os.path.exists(output_layer):
                    os.makedirs(output_layer)
                pkl_list[layer].append([img, os.path.join(output_layer, img_name+".pkl")])


        for layer in layer_list:
            write_cross_pkl(pkl_list[layer], sess, cross_pred)

    if True:
        frame_dir="frame/all/"
        proposal_list = ["faster_bb"]
        pkl_list = {}

        for img_name in os.listdir(frame_dir):
            img_name=img_name.replace(".jpg","").replace(".png","")
            origin_img = cv2.imread( frame_dir+img_name+'.jpg', cv2.IMREAD_COLOR)

            for proposal_dir in proposal_list:
                with open(os.path.join(proposal_dir, img_name+".txt"), 'r') as ff:
                    output_dir = os.path.join(output_root, proposal_dir)
                    proposal_num = 0
                    for linee in ff:
                        token=linee.strip().split()
                        bb=[int(float(token[0])), int(float(token[1])), int(float(token[2])), int(float(token[3]))]
                        box_score=float(token[4])
                        [bb_width,bb_height]=[bb[3]-bb[1],bb[2]-bb[0]]

                        img=origin_img[bb[1]:bb[3],bb[0]:bb[2]]

                        print(bb)
                        img = transform_img(img,227,227)


                        for layer in layer_list:
                            output_layer = os.path.join(output_dir, layer)

                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)

                            key = proposal_dir+"_"+layer
                            if not key in pkl_list:
                                pkl_list[key] = []
                            pkl_list[key].append([img, os.path.join(output_layer, img_name+"_"+str(bb[0])+"_"+str(bb[1])+"_"+str(bb[2])+"_"+str(bb[3])+".pkl")])

                        proposal_num += 1
                        if proposal_num >= proposal_max:
                            break
        for proposal_dir in proposal_list:
            for layer in layer_list:
                key = proposal_dir+"_"+layer
                if not os.path.exists(output_layer):
                    os.makedirs(output_layer)          
                write_pkl(pkl_list[key], sess, pred)