import numpy as np
import cv2
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image
from sklearn.svm import SVC
from SVMclassifier import model as svm
from SVMclassifier import out_encoder


model = load_model('../models/facenet_keras.h5')

# get the face embedding for one face
def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	print(face_pixels.shape)
    # transform face into one sample
    #expand dims adds a new dimension to the tensor
	samples = np.expand_dims(face_pixels, axis=0)
	print(samples.shape)
    # make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]

faceCascade = cv2.CascadeClassifier('C:/Users/alisy/OneDrive/Documents/AIandMachineLearning/FaceNet-FriendScan/Real-Time_Face-Recognition-System/haarcascades/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags= cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces and predict the face name
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #take the face pixels from the frame
        crop_frame = frame[y:y+h, x:x+w]
        #turn the face pixels back into an image
        new_crop = Image.fromarray(crop_frame)
        #resize the image to meet the size requirment of facenet
        new_crop = new_crop.resize((160, 160))
        #turn the image back into a tensor
        crop_frame = np.asarray(new_crop)
        #get the face embedding using the face net model
        face_embed = get_embedding(model, crop_frame)
        #it is a 1d array need to reshape it as a 2d tensor for svm
        face_embed = face_embed.reshape(-1, face_embed.shape[0])
        print(face_embed.shape)
        #predict using our SVM model
        pred = svm.predict(face_embed)
        #get the prediction probabiltiy
        pred_prob = svm.predict_proba(face_embed)
        #pred_prob has probabilities of each class
        print(pred_prob)
        # get name
        class_index = pred[0]
        class_probability = pred_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(pred)
        text = 'Predicted: %s (%.3f%%)' % (predict_names[0], class_probability)
        #add the name to frame but only if the pred is above a certain threshold
        if (class_probability > 60):
            cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
