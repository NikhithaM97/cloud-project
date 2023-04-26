from PIL import Image
import streamlit as st
from functions.CustomFunctions import Cifar10_Prediction

st.header("Image Classification - Pytorch CNN")
st.subheader("Dataset - Cifar 10")

uploaded_file = st.file_uploader("Choose a image ...", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    classes = {1:'automobile', 2:'bird', 3:'cat', 4:'deer', 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    clas = Cifar10_Prediction(image, 'weights/final_model.h5', classes, (32, 32))
    st.header(f"Result: {clas.title()}")

