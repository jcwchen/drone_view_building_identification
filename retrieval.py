#!/usr/bin/python
import os,sys,math,operator,cv2,glob,pickle
import numpy as np
from math import radians, cos, sin, asin, sqrt
import similarity
from similarity import Point
from random import shuffle
import scipy.spatial.distance
import IPython
import ImageDraw 
import ImageColor
import argparse
try:
    from PIL import Image
except:
    import Image
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser()
parser.add_argument('--f', type=str, default="frame/weather/train/", help='retrieval directory')
parser.add_argument('--q', type=str, default="cross", help='query')
parser.add_argument('--r', type=int, default=0, help='retrieved accuracy')


parser.add_argument('--w1', type=float, default=0.315, help='weight')
parser.add_argument('--w2', type=float, default=0.437, help='weight')
parser.add_argument('--w3', type=float, default=0.4, help='weight')

args = parser.parse_args() 
 
def demo(img_name, index, im, bb_array, gt, best_index, output_dir):
    #painting
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    top = 5
    for single_bb in xrange(top):
        if single_bb==0:
            ax.add_patch(
            plt.Rectangle((bb_array[single_bb][0:2])
                           ,bb_array[single_bb][2]-bb_array[single_bb][0]
                           ,bb_array[single_bb][3]-bb_array[single_bb][1]
                           ,fill=False,edgecolor='yellow', linewidth=3.5))
            ax.text(bb_array[single_bb][0] + 6, bb_array[single_bb][1] - 14,
                 '{}, {:.3f}'.format(single_bb+1, float(bb_array[single_bb][4])),
                 bbox=dict(facecolor='yellow', alpha=0.5),
                 fontsize=14, color='black')
        else:
            ax.add_patch(
            plt.Rectangle((bb_array[single_bb][0:2])
                           ,bb_array[single_bb][2]-bb_array[single_bb][0]
                           ,bb_array[single_bb][3]-bb_array[single_bb][1]
                           ,fill=False,edgecolor='blue', linewidth=3.5))
            ax.text(bb_array[single_bb][0] + 6, bb_array[single_bb][1] - 14,
                 '{}, {:.3f}'.format(single_bb+1, float(bb_array[single_bb][4])),
                 bbox=dict(facecolor='blue', alpha=0.5),
                 fontsize=14, color='white')
    ax.add_patch(
    plt.Rectangle((gt[0:2])
               ,gt[2]-gt[0]
               ,gt[3]-gt[1]
               ,fill=False,edgecolor='red', linewidth=3.5))
    """
    ax.add_patch(
    plt.Rectangle((bb_array[best_index][0:2])
                   ,bb_array[best_index][2]-bb_array[best_index][0]
                   ,bb_array[best_index][3]-bb_array[best_index][1]
                   ,fill=False,edgecolor='green', linewidth=3.5))
    ax.text(bb_array[best_index][0] + 6, bb_array[best_index][1] - 14,
         '{:.3f}'.format(float(bb_array[best_index][4])),
         bbox=dict(facecolor='green', alpha=0.5),
         fontsize=14, color='white')
    """
 
    ax.set_title(('{}. {}').format(index, img_name),fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    fig.savefig(os.path.join(output_dir, '{}.jpg'.format(index)),bbox_inches='tight')  
    """
    top=5
    dr = ImageDraw.Draw(im)
    dr.rectangle(gt, outline = "red")
    for single_bb in xrange(top):
        dr.rectangle(bb_array[single_bb][0:4],
        outline = (0, 0, 255*(top-single_bb)/5))
        #im.thumbnail( (400,100) )
    im.save('output/{}.jpg'.format(index))
    """
  
def main():
      
    # setting
    iou_threshold = 0.3
 
    poi_dir = "poi/"
    #frame_dir = "old_frame/"
    #frame_dir = "frame/weather/test/"
    #frame_dir = "frame/location_3/test/"
    #frame_dir = "frame/location/test/"
    #frame_dir = "frame/all/"
    frame_dir = args.f

    #bb_dir = "500_bb_gt/"
    bb_dir = "faster_bb/"
    #bb_dir = "500_bb/"
 
    #search_dir = "search/"
    #streetview_dir = "streetview/"
    #streetview_dir = "streetview_clean/"
    #aerial_dir = "aerial_clean"
    #cross_dir = "cross"
 
    #query_list = [search_dir, streetview_dir]
    #query_list = [search_dir]
    #query_list = [search_dir]
    #query_list = ["image_gt"]
    query_list = [args.q]
    #layer_list = ["pool5", "fc6","fc7"]
    layer_list = ["fc6"]
 
    visual_dir = "visual_feature"
    visual_cnn = "CNN"
    visual_place = "placeCNN"
    visual_faster = "fasterRCNN"
    visual_triplet = "triplet"

 
    #visual_feature_list = [visual_cnn, visual_place, visual_triplet]
    visual_feature_list = [visual_triplet]
 
    output_dir = "output"
 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
 
 
    streetview_num = 1
  
    never = 0
    frame_info = {}
    ans_hit = 0
    img_num = 0
    img_sum = []


  
    #load frame info: drone directeion
    with open("location.txt", 'r') as f:
        for line in f:
            token = line.strip().split("\t")
            frame_info[token[0]] = token[1:5]
            frame_info[token[0]] = [float(i) for i in frame_info[token[0]]]
    # for random many times
    iteration_max = 1
    index = 0
    
    if args.r == -1:
        recall = range(0, 11)
    else:
        recall = [args.r]
    for rr in recall:
        hit = 0
        poi_num = 0

        for iteration in xrange(iteration_max):
            for img_name in os.listdir(frame_dir):
      
                temp = img_name
                im = Image.open(os.path.join(frame_dir, img_name))
      
      
                width, height = im.size
                img_name = img_name.replace(".jpg","").replace(".png","")     
     
                 
                # preload bb deep feature pkl
                bb_proposal = []
                bb_proposal_pkl = {}
                for layer in layer_list:
                    for visual_feature in visual_feature_list:
                        bb_proposal_pkl[visual_feature+"_"+layer] = []
     
                bb_score=[]
                proposal_max = 200
                proposal_num = 0

                with open(bb_dir+img_name+".txt", 'r') as ff:
                    for linee in ff:
                        if proposal_num == proposal_max:
                            break
                        token = linee.strip().split()
                        bb = [int(float(token[0])), int(float(token[1])), int(float(token[2])), int(float(token[3]))]
                        [bb_width,bb_height] = [bb[3]-bb[1],bb[2]-bb[0]]
                        """
                        if bb_height>= height/2 or bb_width>= width/2 or bb_height<= height/50 or bb_width<= width/50:
                            continue
                        """
                        bb_proposal.append(bb)
                        bb_score.append(float(token[4]))
     
                        file_name = img_name+'_'+str(bb[0])+"_"+str(bb[1])+"_"+str(bb[2])+"_"+str(bb[3])+".pkl"
                        for layer in layer_list:
                            for visual_feature in visual_feature_list:
                                pkl_name = os.path.join(visual_dir, visual_feature, bb_dir, layer, file_name)
                                try:
                                    pkl = pickle.load(open(pkl_name,'rb'))
                                except:
                                    IPython.embed()
                                bb_proposal_pkl[visual_feature+"_"+layer].append(pkl.reshape(np.product(pkl.shape)))
                        proposal_num += 1

                #first 
                with open(poi_dir+img_name+".txt", 'r') as f:
                    center_pos = [0,0]
                    [center_pos[0],center_pos[1],yaw,frame_height] = frame_info[img_name]
                    easy_num = 0
     
                    matching_pair=[]
                    ans_gt=[]
                    ans_list=[]
                    ans_index=0
                    img_num += 1

                    gt_list = []

                    for line in f:
                        token = line.strip().split("\t")

                        index+= 1
                        #print(index)
                        #print(img_name)
                        """
                        easy_num+=  1
                        if easy_num>7:
                            break
                        """
                        im = Image.open(frame_dir+temp)
      
                        gps_pos = [0, 0]
                        [name,gps_pos[0],gps_pos[1],google_type,img_ref,gt] = token
                        str_gps_pos = [gps_pos[0],gps_pos[1]]
                        query_name = str(str_gps_pos[0])+'_'+str(str_gps_pos[1])
                          
                        gps_pos[0] = float(gps_pos[0])
                        gps_pos[1] = float(gps_pos[1])
      
      
                        gt = gt.split(',')
                        gt = [float(i) for i in gt]
                        ans_gt.append(gt)
                        gt_list.append([gt, gps_pos, center_pos, yaw])
      
      
                        max_dis = sqrt((height/2)**2+width**2)
                        bb_array = []
                        best_bb = []
                        score = []
      
      
                        #map_middle = [math.tan(1*math.pi/180*similarity.yawtoangle(yaw)),1]
                        #map_vector = [gps_pos[0]-center_pos[0],gps_pos[1]-center_pos[1]]
                        #map_angle = math.degrees(similarity.angle(map_middle,map_vector))
                          
                        map_middle = similarity.yawtoangle(yaw)

                        map_vector = math.degrees(similarity.angle([0,1],
                            similarity.gpsvecotr(Point(gps_pos[1], gps_pos[0]),
                                Point(center_pos[1], center_pos[0]))))
                        
                        #map_vector = math.degrees(similarity.angle([0,1],[gps_pos[1]-center_pos[1],gps_pos[0]-center_pos[0]]))
                        #IPython.embed()

                        if map_vector<0:
                            map_vector = -map_vector
                        else:
                            map_vector = 360-map_vector
      
                        map_angle = map_middle-map_vector
      
                        drone_middle = [0,1]
                         
                        visual_dis_list = []
     
                        for query in query_list:
                            for layer in layer_list:
                                for visual_feature in visual_feature_list:
                                    target_pkl = []
                                    # origin
                                    pkl_name = os.path.join(visual_dir, visual_feature, query, layer, query_name+".pkl")
                                    # for gt testing
                                    #pkl_name = os.path.join(visual_dir, visual_feature, query, layer, img_name+"_"+query_name+".pkl")
     
                                    pkl = pickle.load(open(pkl_name,'rb'))
                                    bb_proposal_array = np.array(bb_proposal_pkl[visual_feature+"_"+layer])
                                    target_pkl.append(pkl.reshape(np.product(pkl.shape)))
                                    target_pkl = np.array(target_pkl)                        
                                    visual_dis_list.append(scipy.spatial.distance.cdist(target_pkl,bb_proposal_array))
     
     

     
      
                        for bb_index in xrange(len(bb_proposal)):
                            bb = bb_proposal[bb_index]
      
                  
      
      
                            #im_crop = im.crop((bb[0],bb[1],bb[2],bb[3]))
                            #data_histo = get_hsv_histo(im_crop)
                            #dis = color_dis(query_histo,data_histo)
      
                            drone_vector = [(bb[0]+bb[2])/2-width/2,height-bb[3]]
                            drone_angle = math.degrees(similarity.angle(drone_middle, drone_vector))
                              
      
      
                            #distance 
                            angle_dis = math.fabs(map_angle-drone_angle)
                            """
                            if angle_dis>45:
                                angle_dis=400
                            """
                            scale_dis = similarity.likelyscale(bb,height,width)
                            




                            iou = similarity.iou(bb, gt) 

                            p1 = Point(gps_pos[1], gps_pos[0])
                            origin_p = Point(center_pos[1], center_pos[0])


                            if iou >= iou_threshold:
                                distance_x_dis = similarity.likelydistance(gps_pos, center_pos, bb, height, width,
                                    similarity.disofpoint2line(p1, origin_p, similarity.yawtovector(yaw)), True,True)
                                distance_y_dis = similarity.likelydistance(gps_pos, center_pos, bb, height, width,
                                    similarity.disofpoint2line(p1, origin_p, similarity.yawtovector(yaw)), False,True)
                            else:
                                distance_x_dis = similarity.likelydistance(gps_pos,center_pos,bb, height, width,
                                    similarity.disofpoint2line(p1, origin_p, similarity.yawtovector(yaw)), True)
                                distance_y_dis = similarity.likelydistance(gps_pos,center_pos,bb, height, width,
                                    similarity.disofpoint2line(p1, origin_p, similarity.yawtovector(yaw)), False)
      
                            visual_dis = []
                            for visual_single in visual_dis_list:
                                visual_dis.append(visual_single[0, bb_index])
     
                            #visual_dis = similarity.likelyvisual_multiple(target_pkl,bb_proposal_pkl[bb_index])
     
                             
                            score.append([angle_dis,
                                         #scale_dis,
                                         distance_x_dis,
                                         distance_y_dis
                                         ]
                                         +[visual_single for visual_single in visual_dis]
                                         )
                             
                             
     
                            bb_array.append([bb[0],bb[1],bb[2],bb[3]])
      
      
                            #bb_array.append([bb[0],bb[1],bb[2],bb[3],scale_dis+angle_dis+distance_dis])
      
      
      
                            #print(str(bb)+"    "+str(angle_dis)+"  "+str(scale_dis)+"  "+str(distance_dis)+"   "+str(visual_dis))
                            #print(str(bb)+"    "+str(visual_dis))
                            #bb_array.append([bb[0],bb[1],bb[2],bb[3],dis+likelyscale(bb,height,width)/5+angle_dis/1000+likelydistance(gps_pos,center_pos,bb,height,width,max_dis)])
                            #print(str(bb)+" "+str(dis)+"   "+str(likelyscale(bb,height,width)/5)+" "+str(angle_dis/1000)+" "+str(likelydistance(gps_pos,center_pos,bb,height,width,max_dis)))
                          
                        #similarity
                        sim_num = len(score[0])
                        sim = []
                        score = np.array(score)
                        for sim_i in xrange(sim_num):
                            sim.append(similarity.dis_to_sim(score[:,sim_i]))
      
                        for single_bb in xrange(len(bb_array)):
                            sim_score = 0
                            for sim_i in xrange(sim_num):
                                 
                                #weight
                                #all 0.31, 0.44, 0.74
                                #single 0.31, 0.28, 0.67
                                #dis 0.28, 0.6
                                if sim_i == 0:
                                    sim_score += args.w1*sim[sim_i][single_bb] #args.w #0.31
                                elif sim_i == 1:
                                    sim_score += args.w2*sim[sim_i][single_bb] #0.28
                                elif sim_i == 2:
                                    sim_score += args.w3*sim[sim_i][single_bb] #0.67
                                else:
                                    sim_score += sim[sim_i][single_bb]
                            
                                # filter angle
                                """
                                if sim[sim_i][single_bb]==0:
                                    sim_score-=10000
                                else:
                                    sim_score+= sim[sim_i][single_bb]
                                """
                                #sim_score += sim[sim_i][single_bb]
                                #print sim[sim_i][single_bb],
                            #print
                            #sim_score += bb_score[single_bb]  # add classification score
                            bb_array[single_bb].append(sim_score)
                         
                        #bb_array.sort(key = similarity.compare,reverse = True) #cos_sim
                         
                        for single_bb in xrange(len(bb_array)):
                            matching_pair.append([ans_index,single_bb,bb_array[single_bb][0:4],bb_array[single_bb][4]])
                        ans_list.append(ans_index)
                        ans_index+=1
     
     
                        bb_array.sort(key = lambda x: float(x[4]),reverse = True) #l2_distance
                        #bb_array.sort(key = lambda x: float(x[4])) #l2_distance
                         
                         
                        """
                        print(bb_array[0][0:4])
                        print("drone_angle: "+str(bb_array[0][5]))
                        print("map_middle: "+str(map_middle))
                        print("map_vector: "+str(map_vector))
                        print("map_angle: "+str(map_angle))
                        print(gps_pos)
                        print(center_pos)
                        print(yaw)
                          
                        im.save('test.png')
                        exit(0)
                        """
      
                        #shuffle(bb_array)
                              
      
                        #print(str(bb_array[0])+" "+str(bb_array[0][4]))
                        #print(bb_array[0][0:4],gt)
      
                        #eval_angle
                        #hit+=  similarity.eval_angle(width,height,bb_array[0][0:4],gt)
                        best_iou = 0
                        for bb_index in xrange(len(bb_array)):   
                            iou = similarity.iou(bb_array[bb_index][0:4],gt)
                            if iou>best_iou:
                                best_iou = iou
                                best_index = bb_index
                        if best_iou < iou_threshold: 
                            never += 1
                            best_iou = 0.01
                            #continue
                        #iou
                        poi_num+= 1
      
                        for bb_index in xrange(len(bb_array)):
                            iou = similarity.iou(bb_array[bb_index][0:4],gt)
                            if iou >= iou_threshold and rr == 0:
                                hit+=  float(1) / (bb_index+1)
                                break                            

                            elif iou >= iou_threshold and bb_index < rr:
                                hit+=  float(1)
                                break
                        #demo
                        #demo(img_name, index, im, bb_array, gt, best_index, output_dir) 
     
                    #gt_list.append(gt, gps_pos, center_pos, yaw)
                    """
                    gt_score = []
                    for single in gt_list:
                        [gt, gps_pos, center_pos, yaw] = single
                        p1 = Point(gps_pos[1], gps_pos[0])
                        origin_p = Point(center_pos[1], center_pos[0])
                        gt_score.append([float(gt[2]+gt[0])/2
                            , similarity.disofpoint2line(p1, origin_p, similarity.yawtovector(yaw))])



                    gt_score.sort(key = lambda x: float(x[1]), reverse = True)

                    dcg_list = [gs[0] for gs in gt_score]
                    print(similarity.DCG(dcg_list))
                    """
                    #IPython.embed()
                    """
                    has_choose=[]
     
                    matching_pair.sort(key = lambda x: float(x[3]),reverse = True)
     
                    ans_length=[1 for i in xrange(len(ans_gt))]
     
     
                    for pair_index in xrange(len(matching_pair)):
                        if matching_pair[pair_index][0] in ans_list and not matching_pair[pair_index][1] in has_choose:                       
                            iou = similarity.iou(matching_pair[pair_index][2], ans_gt[matching_pair[pair_index][0]])
     
                            if iou >= 0.5:
                                has_choose.append(matching_pair[pair_index][1])
                                ans_list.remove(matching_pair[pair_index][0])                            
                                ans_hit+=  float(1) / len(ans_gt)/ans_length[matching_pair[pair_index][0]] 
                         
                            else:
                                ans_length[matching_pair[pair_index][0]]+= 1
                    
     
     
                            if not ans_list:
                                break 
                    # new start
                    img_ans_list = []
                    top_k = 100

                    for start_index in xrange(top_k):

                        has_choose = []
                        ans_list = [i for i in xrange(len(ans_gt))]
                        img_ans = []
                        img_ans_score = 0

                        for pair_index in xrange(start_index, len(matching_pair)):
                            if matching_pair[pair_index][0] in ans_list and not matching_pair[pair_index][1] in has_choose:                       
     
                                has_choose.append(matching_pair[pair_index][1])
                                ans_list.remove(matching_pair[pair_index][0])
                                img_ans_score += matching_pair[pair_index][3]
                                img_ans.append([matching_pair[pair_index][0], matching_pair[pair_index][2]])                        
                                  
                                if not ans_list:
                                    img_ans_list.append([img_ans, img_ans_score])
                                    break

                        img_ans_list.sort(key = lambda x: float(x[1]), reverse = True)
                        
                        after_dcg = []
                        
                        for img_ans in img_ans_list:
                            
                            img_ans, score = img_ans
                            gt_score = []

                            for single in img_ans:

                                [gt, gps_pos, center_pos, yaw] = gt_list[single[0]]
                                bb = single[1]
                                p1 = Point(gps_pos[1], gps_pos[0])
                                origin_p = Point(center_pos[1], center_pos[0])
                                gt_score.append([float(bb[2]+bb[0])/2
                                    , similarity.disofpoint2line(p1, origin_p, similarity.yawtovector(yaw))])

                            gt_score.sort(key = lambda x: float(x[1]), reverse = True)

                            dcg_list = [gs[0] for gs in gt_score]
                            after_dcg.append([img_ans, score+10*similarity.sigmoid(similarity.DCG(dcg_list)/5000)])



                    after_dcg.sort(key = lambda x: float(x[1]), reverse = True)

                    img_ans = after_dcg[0][0]
                    for single in img_ans:
                        #IPython.embed()
                        if similarity.iou(single[1], ans_gt[single[0]]) > 0.5:
                            img_sum.append(1)
                        else:
                            img_sum.append(0)
                    """
                    #IPython.embed()


        #print("-------------------------------------")
        #print(rr)
        #print(hit)
        #print(poi_num)
        print("{:.4f}".format(hit/poi_num))
    print("Whole: {:.4f}".format(ans_hit/img_num))
    #print("multi: {:.4f}".format(np.mean(img_sum)))
    print("# of Img: {}".format(img_num))
    print("# of POI: {}".format(poi_num))
    print("# Never : {}".format(never))
 
    #embed()
    #145 383
if __name__  ==  '__main__':
    main()
 
"""
https://www.google.com.tw/maps/place/25%C2%B001'38.5%22N+121%C2%B032'09.2%22E/@25.0287444,121.5360478,15z/data = !4m5!3m4!1s0x0:0x0!8m2!3d25.027352!4d121.535887
https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=25.0415889,121.5640499&radius=2000&key=AIzaSyDAnTTBRbpMAinjfngLWKQWVW1w-OojBLE&type=establishment&keyword=%E8%87%BA%E5%8C%97%E5%B0%8F%E5%B7%A8%E8%9B%8B
https://maps.googleapis.com/maps/api/streetview?size=600x600&location=25.037525,121.563782&pitch=30&key=AIzaSyDAnTTBRbpMAinjfngLWKQWVW1w-OojBLE
http://www.cmlab.csie.ntu.edu.tw/~jacky82226/demo/retrieval.html 
"""