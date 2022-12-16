import streamlit as st
from PIL import Image 
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from skimage import transform

classes = ['Mild', 'Moderate', 'No DR','PDR', 'Severe']

def loadModel():
    with st.spinner('Model is being loaded..'):
        model=load_model('model')
    return model 
model = loadModel()
def upload_image():
    image_file = st.file_uploader("Upload An Image",type=['png','jpeg','jpg'])
    if image_file is not None:
        st.image(image_file, width=300)         
    return image_file
    
def load():
    np_image = Image.open(upload_image())
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (384, 384, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image

def predict(model):    
    img = load()
    y = model.predict(img)  
    y_preds = np.argmax(y)
    if y_preds == 0:
        st.write("The image is classified as : {}".format(classes[0]))
    elif y_preds == 1:
        st.write("The image is classified as : {}".format(classes[1]))
    elif y_preds == 2:
        st.write("The image is classified as : {}".format(classes[2]))
    elif y_preds == 3:
        st.write("The image is classified as : {}".format(classes[3]))
    elif y_preds == 4:
        st.write("The image is classified as : {}".format(classes[4]))
    

predict(model)
