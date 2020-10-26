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

import re
import sys
import time

from google.cloud import speech
import pyaudio
from six.moves import queue
import os
import playsound
from google.cloud import texttospeech

import subprocess
import inspect

import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import json

import smtplib
from email.mime.text import MIMEText

os.environ['GOOGLE_APPLICATION_CREDENTIALS']='./STT_TTS/speech-to-text-292808-d4d23dae896a.json'

COUNT = 0

## function define ##
def send_mail(text, target_email):
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    
    smtp.ehlo()
    smtp.starttls()
    
    my_email = 'mountaingyu@gmail.com'
    my_email_pw = 'drflvuubntuijpng'
    your_email = target_email
    
    smtp.login(my_email, my_email_pw)
    
    message = MIMEText(text)
    message['Subject'] = '기사 제목'
    message['From'] = my_email
    message['To'] = your_email
    
    smtp.sendmail(my_email, your_email, message.as_string())
    
    smtp.quit()




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

print('Ready to Starting POSVATOR SERVER......')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'poscointernational'
socketio = SocketIO(app)
thread = None
detector = None
model = None

known_face_encodings = None
known_face_names = None


now = datetime.datetime.now() #파일이름 현 시간으로 저장하기
day_before = now - datetime.timedelta(days=1)
now = str(datetime.datetime.now())[:10].replace('-', '')
day_before = str(day_before)[:10].replace('-', '')


def get_news(n_url):
    news_detail = []
    headers = {'User-Agent' : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'}
    breq = requests.get(n_url, headers=headers)
    bsoup = BeautifulSoup(breq.content, 'html.parser')
    
    title = bsoup.select('#articleTitle')[0].text  #대괄호는  h3#articleTitle 인 것중 첫번째 그룹만 가져오겠다.
    news_detail.append(title)
    pdate = bsoup.select('.t11')[0].get_text()[:11]
    news_detail.append(pdate)

    _text = bsoup.select('#articleBodyContents')[0].get_text().replace('\n', " ")
    btext = _text.replace("// flash 오류를 우회하기 위한 함수 추가 function _flash_removeCallback() {}", "")
    news_detail.append(btext.strip())
  
    news_detail.append(n_url)
    
    pcompany = bsoup.select('#footer address')[0].a.get_text()
    news_detail.append(pcompany)

    return news_detail


def crawler(maxpage,query,s_date, e_date):
    s_from = s_date
    e_to = e_date
    page = 1
    maxpage_t =(int(maxpage)-1)*10+1   # 11= 2페이지 21=3페이지 31=4페이지  ...81=9페이지 , 91=10페이지, 101=11페이지
    news = {'title':[], 'contents':[]}
    
    while page < maxpage_t:
        url = "https://search.naver.com/search.naver?where=news&query=" + query + "&sort=1&ds=" + s_date + "&de=" + e_date + "&nso=so%3Ar%2Cp%3Afrom" + s_from + "to" + e_to + "%2Ca%3A&start=" + str(page)
        
        req = requests.get(url)
        cont = req.content
        soup = BeautifulSoup(cont, 'html.parser')
    
        for urls in soup.select("._sp_each_url"):
            try :
                if urls["href"].startswith("https://news.naver.com"):
                    news_detail = get_news(urls["href"])
                        # pdate, pcompany, title, btext
                    news['title'].append(news_detail[0])
                    news['contents'].append(news_detail[2])
            except Exception as e:
                print(e)
                continue
        page += 10
    return news
        
    
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
    
    total_news = dict()
    
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
            known_face_encodings.append(emb)
            infos = file.split('/')[2]
            info = infos.split('_')
            users[info[0]] = [info[1], info[2][:-4], 0]
            
            total_news[info[2][:-4]] = crawler(10, info[2][:-4], day_before, now)
                            
            print(infos)
            known_face_names.append(info[0])
    
    cctv_url = 'http://posproject201013.iptime.org:8787/video_feed'
    video = cv2.VideoCapture(cctv_url)
    while True:
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
                    
                    lab = cv2.cvtColor(croped_img, cv2.COLOR_BGR2LAB)
                    lab_planes = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8 ,8))
                    lab_planes[0] = clahe.apply(lab_planes[0])
                    lab = cv2.merge(lab_planes)
                    croped_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)      
                    
                    
                    emb = img_to_encoding2(croped_img, model)[0]
                    emb = emb.reshape(-1)
                    sim = -1
                    idx = 0
                    for i, known_face in enumerate(known_face_encodings):
                        cur_sim = cos_sim(known_face, emb)
                        if sim < cur_sim:
                            sim = cur_sim
                            idx = i

                    if sim > 0.8:
                        name = known_face_names[idx]

                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(img, name, (box[0] + 6, box[1] - 6), font, 1.0, (255, 255, 255), 1)
                
                if name != 'Mask Detected. Please take off your mask.' and name != 'Unknown':
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
                socketio.emit('message', {'name': user, 'floor' : users[user][0], 'company': users[user][1], 'title':total_news[users[user][1]]['title'], 'content':total_news[users[user][1]]['contents']})
#                 socketio.emit('message', {'name': user, 'floor' : users[user][0], 'company': users[user][1]})
                users[user][2] = 0
        time.sleep(0.01)

        
@socketio.on('email')
def handle_my_custom_event(json):
    print('received json: ' + str(json))        
  
        
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