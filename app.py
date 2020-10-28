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
from RetinaFace.retinaface_cov import RetinaFaceCoV
from PIL import Image, ImageFile
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

import re
import sys
import time

import subprocess
import inspect

import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import json

import smtplib
from email.mime.text import MIMEText

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

## function define ##
def send_mail(text):
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    
    smtp.ehlo()
    smtp.starttls()
    
    my_email = # Fill your Email
    my_email_pw = # Fill your PW
    your_email = text['email']
    
    smtp.login(my_email, my_email_pw)
    
    results = ""
    for i in range(len(text['title'])):
        results += '\n'
        results += '제목 : ' + text['title'][i] + '\n'
        results += '-' * (3 * len(text['title'][i]) + 5)  + '\n'
        results += '내용 : ' + text['content'][i] + '\n'
        results += '\n'
    
    results += '\n'
    message = MIMEText(results)
    message['Subject'] = str(datetime.datetime.now()) + '_기사 모음'
    message['From'] = my_email
    message['To'] = your_email
    
    smtp.sendmail(my_email, your_email, message.as_string())
    
    smtp.quit()


def img_to_encoding2(image, model, path=True):
    img = image
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding


def cos_sim(A, B):
    return dot(A,B) / (norm(A) * norm(B))


## function define ##

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
    
    known_face_encodings = dict()
    
    users = dict()
    files = glob.glob('./users/*.jpg')
    
    total_news = dict()
    news = dict()
    
    tfidf = TfidfVectorizer()
    
    korname2engname = dict()

    for file in files:
        sep = file.split('/')[2]
        r_sep = sep.split('_')
        if r_sep[1] == 'augmentation':
            pass
        else:
            total_news[r_sep[3]] = { 'title' : [] , 'contents' : [], 'summary' : [] }
            known_face_encodings[r_sep[0]] = []
            korname2engname[r_sep[3]] = r_sep[0]
                
    print(known_face_encodings)
    print(korname2engname)
                
    crawled_news = pd.read_csv('reco2.csv')
    
    for i in range(len(crawled_news)):
        user = crawled_news.iloc[i]['user']
        title = crawled_news.iloc[i]['title']
        content = crawled_news.iloc[i]['contents']
        summary = crawled_news.iloc[i]['summary']
        total_news[user]['title'].append(title)
        total_news[user]['contents'].append(content)
        total_news[user]['summary'].append(summary)
    
    
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
            
            lab = cv2.cvtColor(croped_img, cv2.COLOR_BGR2LAB)
            lab_planes = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8 ,8))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            croped_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)            
            
            emb = img_to_encoding2(croped_img, model)[0]
            emb = emb.reshape(-1)
            
            infos = file.split('/')[2]
            info = infos.split('_')
            
            if info[1] == 'augmentation':
                known_face_encodings[korname2engname[info[0]]].append(emb)
                continue
            else:
                known_face_encodings[info[0]].append(emb)
                
            
            # floor, company, count, real_name, email 
            users[info[0]] = [info[1], info[2], 0, info[3], info[4][:-4]]
            
    cctv_url = 'http://posproject201013.iptime.org:8787/video_feed'
    video = cv2.VideoCapture(cctv_url)
    mask_count = 0
    mask_count_thresh = 15
    
    
    
    while True:
        try:
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

            if faces is None:
                continue

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

                        lab = cv2.cvtColor(croped_img, cv2.COLOR_BGR2LAB)
                        lab_planes = cv2.split(lab)
                        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8 ,8))
                        lab_planes[0] = clahe.apply(lab_planes[0])
                        lab = cv2.merge(lab_planes)
                        croped_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)      
                        emb = img_to_encoding2(croped_img, model)[0]
                        emb = emb.reshape(-1)
                        sim = 0.5
                        idx = 0


                        for names in known_face_encodings.keys():
                            for user_emb in known_face_encodings[names]:
                                cur_sim = cos_sim(user_emb, emb)
                                if sim < cur_sim:
                                    sim = cur_sim
                                    name = names

                        if sim < 0.80:
                            name = 'Unknown'

                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(img, name, (box[0] + 6, box[1] - 6), font, 1.0, (255, 255, 255), 1)

                    if name != 'Mask Detected. Please take off your mask.' and name != 'Unknown':
                        print(name, 'detected !!!')
                        users[name][2] += 1
                        if users[name][2] >= 15:
                            print('데이터전송')
                            socketio.emit('message', {'name': users[name][3], 'floor' : users[name][0], 'company': users[name][1], 'title':total_news[users[name][3]]['title'], 'content':total_news[users[name][3]]['summary'],'send':total_news[users[name][3]]['contents'], 'email' : users[name][4]})
                            users[name][2] = 0                        
                    else:
                        if name == 'Mask Detected. Please take off your mask.':
                            mask_count += 1
                            if mask_count >= mask_count_thresh:
                                mask_count_thresh += mask_count * 2
                                mask_count = 0
                                print('음성 재생 필요!')
                                requests.get(url = 'http://posproject201013.iptime.org:8787/take_mask')

                        print('no one detected.')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ret, jpeg = cv2.imencode('.jpg', img)
            # print("after get_frame")
            if img is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except Exception as e:
            print('error occured', e)
        time.sleep(0.01)

        
@socketio.on('email')
def handle_my_custom_event(json):
    print('received json: ' + str(json))
    send_mail(json)
    
    
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/")
def index():
    return render_template("index.html", data=None  )


@socketio.on('connect')
def connect():
    print('client connected!')


    
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8999, debug=True)
