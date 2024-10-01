import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List,Tuple

#prevents machine from sucking up all memory and get out of memory errors
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0],True)
except:
  pass

def load_video(path: str) -> tf.Tensor:

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        frame = tf.image.rgb_to_grayscale(frame)
        cropped_frame = frame[190:236,80:220,:]    
        frames.append(cropped_frame)

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames were captured from the video.")
    
    frames_tensor = tf.stack(frames)
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)
    std = tf.maximum(std, 1e-8)
    return tf.cast((frames_tensor - mean), tf.float32) / std

#creating vocab- every single character that is expected to be encountered in our annotations
vocab= [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

#originally from the keras ctc Automatic Speech Recognition tutorial paper
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="") #converts char to num
num_to_char = tf.keras.layers.StringLookup(vocabulary= char_to_num.get_vocabulary(), oov_token="", invert=True)

#load up our alignments
def load_alignments(path:str) -> List[str]:

    with open(path, 'r') as f:
        lines = f.readlines()

    tokens = []

    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens,' ',line[2]]
    tokens_tensor = tf.strings.unicode_split(tokens, input_encoding='UTF-8')
    tokens_reshaped = tf.reshape(tokens_tensor, (-1))

    return char_to_num(tokens_reshaped)[1:]

#returning the preprocessed videos and preprocessed alignments together from their respective paths
def load_data(path: str) -> Tuple[tf.Tensor, tf.Tensor]:
  
  path_str = path.numpy().decode('utf-8')
  filename = os.path.splitext(os.path.basename(path_str))[0]
  video_path = os.path.join('data', 's1', f'{filename}.mpg')
  alignment_path = os.path.join('data', 'alignments', 's1', f'{filename}.align')
  frames=load_video(video_path)
  alignments = load_alignments(alignment_path)
  return frames, alignments

def mappable_function(path:str)->Tuple[tf.Tensor, tf.Tensor]:
  result=tf.py_function(load_data, [path], (tf.float32,tf.int64))
  result[0].set_shape([75, None, None, None])  # Shape for video frames
  result[1].set_shape([40]) 
  return result
  