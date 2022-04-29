import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding' , False )

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('plant_predt.h5')
    return model

model = load_model()


st.title("welcome to plant disease prediction")

file = st.file_uploader("Please upload an image Of Cotton plant to classify", type = ["jpg" ,"png"])
from PIL import Image,ImageOps
import numpy as np
import cv2

def import_and_predict(image_data , model):
    size = (155 , 155)
    image = ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("Please upload a valid image file")

else:
    image = Image.open(file)
    st.image(image , use_column_width = True)
    predictions = import_and_predict(image , model)
    class_names = ['diseased cotton leaf','diseased cotton plant','fresh cotton leaf','fresh cotton plant']
    string = "The image is a " +class_names[np.argmax(predictions)]
    st.success(string)
