import tensorflow as tf
from typing import List
import cv2
import os 
import imageio

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

def load_video(path: str) -> tf.Tensor:

    cap = cv2.VideoCapture(path)
    frames = []

    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        frame = tf.convert_to_tensor(frame, dtype=tf.float32)
        frame = tf.image.rgb_to_grayscale(frame)
        cropped_frame = frame[190:236,80:220,:]
        frames.append(cropped_frame)  

    cap.release()
    frames_tensor = tf.stack(frames)
    mean = tf.math.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)
    std = tf.maximum(std, 1e-8)
    return tf.cast((frames_tensor - mean), tf.float32) / std

    
def load_alignments(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_data(path:str):
  path = bytes.decode(path.numpy())
  filename = path.split('\\')[-1].split('.')[0]
  video_path = os.path.join('..','data','s1',f'{filename}.mpg')
  alignment_path = os.path.join('..','data','alignments','s1',f'{filename}.align')
  frames=load_video(video_path)
  alignments = load_alignments(alignment_path)

  return frames, alignments