import os
import glob

import streamlit as st
from main import predict
from PIL import Image


# all models list
models_list = [
            "vgg16",
            "vgg19",
            "inception",
            "xception",
            "resnet"
        ]

# all images in 'images' folder
images = [image.split("\\")[1] for image in glob.glob(os.path.join("images", "*.jpg"))]

st.sidebar.title("ImageNet")
selected_image = st.sidebar.selectbox("Pick an image.", images)

selected_model = st.sidebar.selectbox("Pick a model.", models_list)

st.write("Enjoy Machine Learning with Streamlit !!!")

if st.sidebar.button('Predict'):
    showpred = 1
    prediction, prob = predict(
                                    selected_model,
                                    os.path.join("images", selected_image)
                                )

    image = Image.open(os.path.join("images", selected_image))
    # image = image.resize((128, 128))
    # st.write(prediction)
    st.image(image,
        caption=f"prediction: {prediction}, probability: {prob * 100}",
        use_column_width=True
        )