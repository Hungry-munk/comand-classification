import tensorflow as tf
import numpy as np
from configs import Configs as C

# global variables
c = C()
num_classes = c.num_classes

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
    # come back later to .audio
    # wav = wav.audio
    # Resample to 16kHz if necessary
    if sample_rate != target_rate:
        wav = resample_wav(wav, sample_rate, target_rate)

    if wav.shape[0] < target_rate:
        target_length = target_rate * 1
        end_padding = target_length - tf.shape(wav)[0]
        paddings = [[0, end_padding], [0,0]]
        wav = tf.pad(wav, paddings, mode='constant', constant_values=0)

    # Convert to mono by taking the first channel if stereo
    if wav.shape[-1] > 1 :
        wav = convert_to_mono(wav)
    # pad the wav to 1 sec


    return wav
def wav_to_spectrogram(wav_tensor, frame_length=255, frame_step=128):
    if len(wav_tensor.shape) > 1:
        wav_tensor = tf.squeeze(wav_tensor)
    # Compute the Short-time Fourier Transform (STFT)
    stft = tf.signal.stft(
        signals=wav_tensor, 
        frame_length=frame_length, 
        frame_step=frame_step
    )
    
    # Compute the magnitude of the STFT
    spectrogram = tf.abs(stft)
    
    return spectrogram

# A function that is going to take in a set of file paths and retrun the appriopriate X and Y object for that data
def batch_generator(X_file_paths, batch_size, target_rate, frame_length, frame_step):
    x_data = []
    y_data = []

    # trackers
    data_example_count = len(X_file_paths)
    batch_length_count = 0
    for x_path in X_file_paths:
        # iterate thorugh files and create a data set for using those file paths
        
        # get label for path by splitting and indexing into secound item (first is base dir)
        label = x_path.split('/')[2]
        # create one hot encoding for softmax
        one_hot = np.zeros(num_classes)
        one_hot_index = c.label_encodings[label]
        one_hot[one_hot_index] = 1
        # get wav audio
        wav = prepare_wav(x_path, target_rate)
        # prepare spectrogram
        spectrogram = wav_to_spectrogram(wav, frame_length, frame_step)
        # logormithic normalziation of spectrogram
        spectrogram = tf.cast(spectrogram, tf.float32)
        spectrogram = tf.math.log1p(spectrogram) #log1p is a better version of log + epsillon

        # add data to dataset
        x_data.append(spectrogram)
        y_data.append(one_hot)
        batch_length_count += 1

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
            data_example_count -= batch_size
        # When their is not data for a full batch 
        elif data_example_count < batch_size :
            return np.array(x_data), np.array(y_data)


def calculate_spectrogram_dimensions(audio_time, target_rate, frame_length, frame_step):
    audio_length = audio_time * target_rate
    # Calculate number of frames
    num_frames = 1 + (audio_length - frame_length) // frame_step
    
    # Number of frequency bins is half the frame length + 1
    num_frequency_bins = (frame_length // 2) + 1
    
    return (num_frames, num_frequency_bins)

#A function to create tensorflow data set for better GPU acceleration during training, validation and testing
def create_dataset(X_file_paths, batch_size, target_rate, frame_length, frame_step  ):
    # calcualte  spectrogram height and width
    audio_time = 1
    height, width = calculate_spectrogram_dimensions(audio_time, target_rate, frame_length, frame_step)
    
    dataset = tf.data.Dataset.from_generator(
        lambda: batch_generator(X_file_paths, batch_size, target_rate, frame_length, frame_step),
        output_signature = (
            tf.TensorSpec(shape=(batch_size, height, width), dtype= tf.float32), #define the shape of X's and Y's
            tf.TensorSpec(shape=(batch_size, num_classes), dtype=tf.int32)
        )
    )

    return dataset
