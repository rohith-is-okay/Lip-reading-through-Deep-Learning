# Import necessary libraries
import streamlit as st
import os 
import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
import imageio
import numpy as np
from PIL import Image
import moviepy.editor as mp

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

# Main content
st.title('LipSync Studio') 


# Function to convert .mpg to .mp4
def convert_mpg_to_mp4(mpg_path, mp4_path):
    clip = mp.VideoFileClip(mpg_path)
    clip.write_videofile(mp4_path, codec="libx264")

# Upload video and display
options = os.listdir(os.path.join('..','data','s1'))
selected_video = st.selectbox('Upload a video',options)
if options: 
    col1, col2 = st.columns(2)
    with col1: 
        st.info('Uploaded video:')
        file_path = os.path.join('..','data','s1',selected_video)
        if selected_video.endswith('.mpg'):
            mp4_path = file_path.replace('.mpg', '.mp4')
            
            # Convert only if the .mp4 version doesn't exist
            if not os.path.exists(mp4_path):
                convert_mpg_to_mp4(file_path, mp4_path)
            
            file_path = mp4_path
        video = open(file_path, 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)

    with col2: 
        st.info('This is representative of what the ML model sees when making a prediction. The video has been preprocessed to have zero mean and unit standard deviation.')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        images = []
        for frame in video:
            frame_unit = np.uint8(frame * 255)  # Scale pixel values to 0-255
            frame_squeeze = np.squeeze(frame_unit)  # Remove any single-dimensional entries

            if frame_squeeze.ndim == 2:  # Grayscale to RGB
                frame_squeeze = np.stack((frame_squeeze,) * 3, axis=-1)

            img = Image.fromarray(frame_squeeze)
            images.append(img)

        if images:
            imageio.mimsave('animation.gif', images, fps=10, subrectangles=True, palettesize=256, quantizer="nq")
            st.image('animation.gif', width=350)
        else:
            st.warning('No frames available to create GIF. Please check the video processing step.')


st.info('This is the output of the machine learning model as tokens')
model = load_model()
video, annotations = load_data(tf.convert_to_tensor(file_path))
yhat = model.predict(tf.expand_dims(video, axis=0))
decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
st.text(decoder)



st.info('SUCCESS!')
converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

st.markdown("Text generated from the video is : "+f'<span style="color: green;">{converted_prediction}</span>', unsafe_allow_html=True)
