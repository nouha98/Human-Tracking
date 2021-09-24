from absl import flags
import sys
FLAGS = flags.FLAGS
FLAGS(sys.argv)
# for initilaising flags setting  for yolo v3

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime 
from datetime import date 

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images # resising img
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet # features gen ecoding

def markAttendance(ID):
    with open('Attendance.csv', 'r+') as f:
        data = f.readlines()
        IDs = []
        for line in data:
            entry = line.split(',')
            IDs.append(entry[0])
        if ID not in IDs:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            todays_date= date.today() 
            f.writelines(f'\n{ID},{dtString},{todays_date}')

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

#for considering if the objects same/not in the prev/curr frame # if >0.5  -> similaire 
max_cosine_distance = 0.5 
# create gallery and store the feat letters   (dnn extract feat ) def 100
nn_budget = None  
# to avoid if there are too many det of the similaire obj
nms_max_overlap = 0.8

#pre-trained CNN for tracking
model_filename = 'model_data/mars-small128.pb'

#feature generation 
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
#returns the closest distance to any sample that has been observed so far /metric : "euclidean" or "cosine"
# matching_threshold :max_cosine_distance Samples with larger distance are considered an invalid match

tracker = Tracker(metric)

vid = cv2.VideoCapture(0)
#saving vid
#output as AVI(Audio Video Interleave)
codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS)) #frame rate
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./data/video/r1.avi', codec, vid_fps, (vid_width, vid_height))

from _collections import deque
# deque preferred over list  quicker append and pop operations o(1)

pts = [deque(maxlen=30) for _ in range(1000)]

counter = []

while True:
    _, img = vid.read()
    if img is None:
        print('Completed')
        break
#transf in order to put it in yolo
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0) # add dim : batch size /height width+depth 
    img_in = transform_images(img_in, 416)  #resize 

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)

#boxes,3D shape (1,100,4) : xy centre coord +widh+heighy
# scores (confidence )/classes 2D
#nums the tot nbr of detected obj
    classes = classes[0]
    names = []  #  names ofOD 
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)

    #remove the 0 and scale it according to the orig size of 
    converted_boxes = convert_boxes(img, boxes[0])
    #gen features for each of obj dete
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

#perform suppression on the detection frame to eliminate multiple frames on one target
   
#tlwh(top-left-width-height
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    #whih obj is diregarded
    detections = [detections[i] for i in indices]

    tracker.predict() # based on Kalman  filltering
    #update command tracker param and feature set
    tracker.update(detections)
# ) vis the result
    cmap = plt.get_cmap('tab20b')  # generate color maps (nvr to col)

    #gen 20 col 
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    current_count = int(0)

    for track in tracker.tracks:
        #if fil could notassign track + no upd for the track -> skip
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()

        #colors when tracking
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)

        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

        height, width, _ = img.shape
        cv2.line(img, (int(width /2),0), (int(width /2),height), (0, 255, 0), thickness=2) 

        center_y = int(((bbox[1])+(bbox[3]))/2)

        if class_name == 'person' :  
            current_count += 1
            if center_y <= int(width /2) : 
                counter.append(int(track.track_id))
                markAttendance(str(track.track_id)) 
                

    total_count = len(set(counter))
    cv2.putText(img, "Current person Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
    cv2.putText(img, "Total persons  Count: " + str(total_count), (0,130), 0, 1, (0,0,255), 2)

    fps = 1./(time.time()-t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    cv2.resizeWindow('output', 1024, 768)
    cv2.imshow('output', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()