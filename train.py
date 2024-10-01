import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Conv3D,LSTM, Dropout,Bidirectional, Activation, SpatialDropout3D, BatchNormalization, MaxPool3D,Reshape, Flatten,TimeDistributed
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from utils import num_to_char, char_to_num
from data_pipeline import train, test, data
from model import build_model

input_shape=data.as_numpy_iterator().next()[0][0].shape
vocabulary_size=char_to_num.vocabulary_size()
model = build_model(input_shape, vocabulary_size)

def scheduler(epoch,lr):
  if epoch<30:
    return lr
  else:
    return lr*tf.math.exp(-0.1)

#CTC LOSS - this block of code has been picked up from Automatic Speech Recognition using CTC paper.
def CTCLoss(y_true,y_pred):
  batch_len = tf.cast(tf.shape(y_true)[0], dtype='int64')
  input_length = tf.cast(tf.shape(y_pred)[1], dtype='int64')
  label_length = tf.cast(tf.shape(y_true)[1], dtype='int64')

  input_length=input_length* tf.ones(shape=(batch_len,1),dtype='int64')
  label_length = label_length* tf.ones(shape=(batch_len,1), dtype='int64')

  loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
  return loss

class ProduceExample(tf.keras.callbacks.Callback):
    def __init__(self, dataset) -> None:
        self.dataset = dataset.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs=None) -> None:
        data = self.dataset.next()
        yhat = self.model.predict(data[0])
        decoded = tf.keras.backend.ctc_decode(yhat, [75,75], greedy=False)[0][0].numpy()
        for x in range(len(yhat)):
            print('Original:', tf.strings.reduce_join(num_to_char(data[1][x])).numpy().decode('utf-8'))
            print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            print('~'*100)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),loss =CTCLoss)
checkpoint_callback= ModelCheckpoint(os.path.join('models','checkpoint.h5'),monitor='loss', save_weights_only=True, save_best_only=True)
schedule_callback = LearningRateScheduler(scheduler)
example_callback = ProduceExample(data)
model.fit(train, validation_data=test, epochs=50, callbacks=[checkpoint_callback, schedule_callback, example_callback], )
