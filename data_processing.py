import os
import sys
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from configs import Configs as C
import math

# converting stereo audio into mono for 1d convelution and training
def convert_to_mono(wav):
    return tf.reduce_mean(wav, axis = -1, keepdims= True)

def resample_wav(wav, og_sample_rate, target_sample_rate):
        # Calculate the number of samples needed for 16kHz
        duration = tf.shape(wav)[0] / og_sample_rate
        new_sample_count = tf.cast(duration * target_sample_rate, tf.int32)
        
        # Resample using tf.image.resize with 1D signal
        resampled = tf.image.resize(
            tf.expand_dims(wav, -1), #image representation
            [new_sample_count, 1], 
            method='bilinear'
        )
        
        return tf.squeeze(resampled)

# A function to take in a file path and load and prepare an audio file
def prepare_wav(file_path, target_rate):
    # load audio file in
    audio_binary = tf.io.read_file(file_path)
    wav, sample_rate = tf.audio.decode_wav(audio_binary)

    # Resample to 16kHz if necessary
    if sample_rate != target_rate:
        wav = resample_wav(wav, sample_rate, target_rate)

    # Convert to mono by taking the first channel if stereo
    if wav.shape[-1] > 1 :
        wav = convert_to_mono(wav)

    return wav

def wav_to_spectrogram(wav_tensor, nfft, window, stride):
    return tfio.audio.spectrogram(
        wav_tensor, 
        nfft,
        window,
        stride
    )

# A function that is going to take in a set of file paths and retrun the appriopriate X and Y object for that data
def batch_generator(X_file_paths, batch_size, target_rate, nfft, window, stride):
    x_data = []
    y_data = []

    batch_length_count = 0
    for x_path in X_file_paths:
        # iterate thorugh files and create a data set for using those file paths
        
        # get label for path by splitting and indexing into secound item (first is base dir)
        label = x_path.split(os.path.sep)[1]
        # create one hot encoding for softmax
        one_hot = np.ones(C.num_classes)
        one_hot_index = C.label_encodings[label]
        one_hot[one_hot_index] = 1

        # prepare spectrogram
        wav = prepare_wav(x_path, target_rate)
        spectrogram = wav_to_spectrogram(wav, nfft, window, stride)
        # logormithic normalziation of spectrogram
        spectrogram = tf.cast(spectrogram, tf.float32)
        spectrogram = tf.math.log1p(spectrogram) #log1p is a better version of log + epsillon

        # add data to dataset
        batch_length_count += 1
        x_data.append(spectrogram)
        y_data.append(one_hot)

        if batch_length_count >= batch_size:
            # pre pare and yield batch
            batched_x_data = np.array(x_data[:batch_size])
            batched_y_data = np.array(y_data[:batch_size])
            yield batched_x_data, batched_y_data
            # remove yielded data
            x_data = x_data[batch_size:]
            y_data = y_data[batch_size:]
            # reduce counter
            batch_length_count -= batch_size

def calculate_spectrogram_dimensions(target_rate, nfft, window, stride):
    height = (nfft // 2) + 1

    # Calculate width (time frames)
    width = math.ceil((target_rate - window) / stride) + 1

    return height, width 

#A function to create tensorflow data set for better GPU acceleration during training, validation and testing
def create_dataset(X_file_paths, batch_size, target_rate, nfft, window, stride  ):
    num_classes = C.num_classes
    # calcualte  spectrogram height and width
    height, width = calculate_spectrogram_dimensions(target_rate, nfft, window, stride)
    
    dataset = tf.data.Dataset.from_generator(
        lambda: batch_generator(X_file_paths, batch_size, target_rate, nfft, window, stride ,),
        output_signature = (
            tf.TensorSpec(shape=(batch_size, height, width, 1 ), dtype= tf.float32), #define the shape of X's and Y's
            tf.TensorSpec(shape=(batch_size, num_classes), dtype=tf.int32)
        )
    )

    return dataset
