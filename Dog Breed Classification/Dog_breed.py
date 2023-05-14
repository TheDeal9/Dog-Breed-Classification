import tensorflow as tf
import cv2 
import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model

model=load_model("C:\\Users\\limba\\OneDrive\\Documents\\Dog_Breed\\dog_breed.h5")
class_names=['scottish_deerhound','maltese_dog','bernese_mountain_dog']
st.title('Dog Breed Prediction')
st.markdown('Upload Dog Image')
image=st.file_uploader('Upload Image')
button=st.button('Predict')
if button:
    if image is not None:
        file_bytes=np.asarray(bytearray(image.read()),dtype=np.uint8)
        opencv_image=cv2.imdecode(file_bytes,1)
        
        
        st.image(opencv_image,channels='BGR')
        opencv_image=cv2.resize(opencv_image,(224,224))
        opencv_image.shape=(1,224,224,3)
        Y_pred=model.predict(opencv_image)
        
        st.title(str("The Dog Breed is "+class_names[np.argmax(Y_pred)]))