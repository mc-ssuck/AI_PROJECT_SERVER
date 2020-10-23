from flask import Flask, render_template, Response, redirect, request, session
import cv2
import time
import requests
from flask_socketio import SocketIO, emit
import os
import threading
import tts_stt
import base64

import numpy as np
import glob
import argparse
import dlib
import os
from utils.aux_functions import *
import math
import datetime
from RetinaFace.retinaface_cov import RetinaFaceCoV
from PIL import Image, ImageFile
import insightface
import tensorflow as tf
from parameters import *
import keras

from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle

from numpy import dot
from numpy.linalg import norm

import loadmodel

## function define ##

def verify(image_path, identity, database, model):
    encoding = img_to_encoding2(image_path, model, False)
    min_dist = 1000
    for  pic in database:
        dist = np.linalg.norm(encoding - pic)
        if dist < min_dist:
            min_dist = dist
    print(identity + ' : ' +str(min_dist)+ ' ' + str(len(database)))
    
    if min_dist<THRESHOLD:
        door_open = True
    else:
        door_open = False
    return min_dist, door_open


def img_to_encoding2(image, model, path=True):
    img = image
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


def cos_sim(A, B):
    return dot(A,B) / (norm(A) * norm(B))

## function define ##


cctv_url = 'http://posproject201013.iptime.org:8787/video_feed'

video = cv2.VideoCapture(cctv_url)
print('Ready to Starting POSVATOR SERVER......')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'poscointernational'
socketio = SocketIO(app)
thread = None
detector = None
model = None

known_face_encodings = None
known_face_names = None

def gen():
    global detector, model, known_face_encodings, known_face_names
    """Video streaming generator function."""
    
    detector, model = loadmodel.load_models()
    
    ## extract embeddings from databases ##
    thresh = 0.8
    mask_thresh = 0.2
    known_face_encodings = []
    known_face_names = []
    users = dict()
    files = glob.glob('../users/*.jpg')

    for file in files:
        scales = [640, 1080]
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)

        im_shape = img.shape
        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        scales = [im_scale]
        flip = False

        faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)

        if len(faces) != 0:
            box = faces[0][0:4].astype(np.int)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
            croped_img = img[box[1]:box[3], box[0]:box[2]]
            croped_img = cv2.resize(croped_img, (160, 160), cv2.INTER_CUBIC)
            emb = img_to_encoding2(croped_img, model)[0]
            emb = emb.reshape(-1)
            known_face_encodings.append(emb)
            infos = file.split('/')[2]
            info = infos.split('_')
            users[info[0]] = [info[1], info[2], 0]
            print(infos)
            known_face_names.append(info[0])
    
    while True:
        print('im current reading video from ' + cctv_url)
        scales = [640, 1080]
        _, img = video.read()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        im_shape = img.shape

        target_size = scales[0]
        max_size = scales[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        #im_scale = 1.0
        #if im_size_min>target_size or im_size_max>max_size:
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)

        scales = [im_scale]
        flip = False

        faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
        if len(faces) != 0:
            for i in range(faces.shape[0]):
                face = faces[i]
                box = face[0:4].astype(np.int)
                mask = face[5]

                name = "Unknown"

                if mask>=mask_thresh:
                    name = 'Mask Detected. Please take off your mask.'
                    ###      R G B
                    color = (0,0,255)
                else:
                    color = (0,255,0)
                    croped_img = img[box[1]:box[3], box[0]:box[2]]
                    croped_img = cv2.resize(croped_img, (160, 160), cv2.INTER_CUBIC)
                    emb = img_to_encoding2(croped_img, model)[0]
                    emb = emb.reshape(-1)
                    sim = -1
                    idx = 0
                    for i, known_face in enumerate(known_face_encodings):
                        cur_sim = cos_sim(known_face, emb)
                        if sim < cur_sim:
                            sim = cur_sim
                            idx = i

                    if sim > 0.5:
                        name = known_face_names[idx]

                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (box[0] + 6, box[1] - 6), font, 1.0, (255, 255, 255), 1)
                
                if name != 'Mask Detected. Please take off your mask.':
                    print(name, 'detected !!!')
                    users[name][2] += 1
                else:
                    print('no one detected.')
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', img)
        # print("after get_frame")
        if img is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
        for user in users.keys():
            if users[user][2] >= 10:
                socketio.emit('message', {'name': user, 'floor' : users[user][1], 'company': users[user][2]})
                users[user][2] = 0
        time.sleep(0.01)

        
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/")
def index():
    return render_template("index.html", data=None  )


def image_processing_task():
    global detector, model, known_face_encodings, known_face_names
    
#     detector, model = loadmodel.load_models()
    
#     ## extract embeddings from databases ##
#     thresh = 0.8
#     mask_thresh = 0.2
#     known_face_encodings = []
#     known_face_names = []
#     users = dict()
#     files = glob.glob('../users/*.jpg')

#     for file in files:
#         scales = [640, 1080]
#         img = cv2.imread(file)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)

#         im_shape = img.shape
#         target_size = scales[0]
#         max_size = scales[1]
#         im_size_min = np.min(im_shape[0:2])
#         im_size_max = np.max(im_shape[0:2])
#         im_scale = float(target_size) / float(im_size_min)
        
#         if np.round(im_scale * im_size_max) > max_size:
#             im_scale = float(max_size) / float(im_size_max)

#         scales = [im_scale]
#         flip = False

#         faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)

#         if len(faces) != 0:
#             box = faces[0][0:4].astype(np.int)
#             cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 2)
#             croped_img = img[box[1]:box[3], box[0]:box[2]]
#             croped_img = cv2.resize(croped_img, (160, 160), cv2.INTER_CUBIC)
#             emb = img_to_encoding2(croped_img, model)[0]
#             emb = emb.reshape(-1)
#             known_face_encodings.append(emb)
#             infos = file.split('/')[2]
#             info = infos.split('_')
#             users[info[0]] = [info[1], info[2], 0]
#             print(infos)
#             known_face_names.append(file.split('/')[2][:-4])
    ## extract embeddings from databases ##
    
#     while True:
#         print('im current reading video from ' + cctv_url)
#         scales = [640, 1080]
#         _, img = video.read()

#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         im_shape = img.shape

#         target_size = scales[0]
#         max_size = scales[1]
#         im_size_min = np.min(im_shape[0:2])
#         im_size_max = np.max(im_shape[0:2])
#         #im_scale = 1.0
#         #if im_size_min>target_size or im_size_max>max_size:
#         im_scale = float(target_size) / float(im_size_min)
#         # prevent bigger axis from being more than max_size:
#         if np.round(im_scale * im_size_max) > max_size:
#             im_scale = float(max_size) / float(im_size_max)

#         scales = [im_scale]
#         flip = False

#         faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
#         if len(faces) != 0:
#             for i in range(faces.shape[0]):
#                 face = faces[i]
#                 box = face[0:4].astype(np.int)
#                 mask = face[5]

#                 name = "Unknown"

#                 if mask>=mask_thresh:
#                     name = 'Mask Detected. Please take off your mask.'
#                     ###      R G B
#                     color = (0,0,255)
#                 else:
#                     color = (0,255,0)
#                     croped_img = img[box[1]:box[3], box[0]:box[2]]
#                     croped_img = cv2.resize(croped_img, (160, 160), cv2.INTER_CUBIC)
#                     emb = img_to_encoding2(croped_img, model)[0]
#                     emb = emb.reshape(-1)
#                     sim = -1
#                     idx = 0
#                     for i, known_face in enumerate(known_face_encodings):
#                         cur_sim = cos_sim(known_face, emb)
#                         if sim < cur_sim:
#                             sim = cur_sim
#                             idx = i

#                     if sim > 0.5:
#                         name = known_face_names[idx]

#                 cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
#                 font = cv2.FONT_HERSHEY_DUPLEX
#                 cv2.putText(img, name, (box[0] + 6, box[1] - 6), font, 1.0, (255, 255, 255), 1)
                
#                 if name != 'Mask Detected. Please take off your mask.':
#                     print(name, 'detected !!!')
#                     socketio.emit('message', {'values': name})
#                 else:
#                     print('no one detected.')
#         time.sleep(0.2)


@socketio.on('connect')
def connect():
    print('client connected!')
#     global thread
#     if thread is None:
#         thread = socketio.start_background_task(target=image_processing_task)

        
if __name__ == '__main__':
        
    socketio.run(app, host='0.0.0.0', port=8999, debug=True)