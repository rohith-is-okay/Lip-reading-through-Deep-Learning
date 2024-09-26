# Import necessary libraries
import streamlit as st
import os 
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
import imageio
import numpy as np
from PIL import Image
import pandas as pd

# Define CSS styles
main_bg_color = "#fffff"
main_txt_color = "#000000"
accent_color = "#00001a"

# Apply CSS styles
st.markdown(
    f"""
    <style>
        
    
        .sidebar .sidebar-content {{
            background-color: {accent_color};
            color: {main_txt_color};
        }}
        .sidebar .sidebar-content .block-container {{
            background-color: transparent;
        }}
        .Widget>label {{
            color: {main_txt_color};
        }}
        .st-bw {{
            background-color: {main_bg_color};
        }}
        .st-c3 .css-1v1l6e3 {{
            background-color: {accent_color};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar content
with st.sidebar: 
    st.image("https://cdn.pixabay.com/photo/2013/07/12/18/17/equalizer-153212_1280.png", width=150)
    st.title('LipSync Studio')
    st.info('This is made by using Deep Learning with help of CNN and RNN algorithms.')
    st.info('PARTH & ROHITH')

st.header('Neural Architecture', divider='blue')

st.image('C:/Users/rohit/OneDrive/Desktop/LIPBUDDIES/Slide.png')


data = {
    "Layer (type)": ["conv3d_3 (Conv3D)", "activation_3 (Activation)", "max_pooling3d_3 (MaxPooling3D)",
                     "conv3d_4 (Conv3D)", "activation_4 (Activation)", "max_pooling3d_4 (MaxPooling3D)",
                     "conv3d_5 (Conv3D)", "activation_5 (Activation)", "max_pooling3d_5 (MaxPooling3D)",
                     "time_distributed_1 (TimeDistributed)", "bidirectional_LSTM (Bidirectional)",
                     "dropout_2 (Dropout)", "bidirectional_LSTM (Bidirectional)", "dropout_3 (Dropout)",
                     "dense (Dense)"],
    "Output Shape": ["(None, 75, 46, 140, 128)", "(None, 75, 46, 140, 128)", "(None, 75, 23, 70, 128)",
                     "(None, 75, 23, 70, 256)", "(None, 75, 23, 70, 256)", "(None, 75, 11, 35, 256)",
                     "(None, 75, 11, 35, 75)", "(None, 75, 11, 35, 75)", "(None, 75, 5, 17, 75)",
                     "(None, 75, 6375)", "(None, 75, 256)", "(None, 75, 256)", "(None, 75, 256)",
                     "(None, 75, 256)", "(None, 75, 41)"],
    "Param #":      ["3584", "0", "0", "884992", "0", "0", "518475", "0", "0", "0", "6660096", "0", "394240",
                     "0", "10537"],
}

df = pd.DataFrame(data)

st.table(df)

total_params = "8471924"
trainable_params = "8471924"
non_trainable_params = "0"

st.header("Model Parameters",divider='blue')
st.write("Total params:", total_params, "(", f"{int(total_params) / (1024 * 1024):.2f}", "MB)")
st.write("Trainable params:", trainable_params, "(", f"{int(trainable_params) / (1024 * 1024):.2f}", "MB)")
st.write("Non-trainable params:", non_trainable_params, "(", f"{int(non_trainable_params) / (1024 * 1024):.2f}", "Byte)")


