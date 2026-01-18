import tensorflow as tf
from tensorflow.keras import layers, Model

import pdb

class AdDetectorNN(Model):

    def __init__(self, video_frame_shape, audio_frame_shape, num_classes):
        super(AdDetectorNN, self).__init__()

        self.videoNN = self._video_neural_net_architecture(video_frame_shape)
        self.audioNN = self._audio_neural_net_architecture(audio_frame_shape)
        self.final_layer = layers.Dense(units=num_classes, activation='sigmoid', name='final_layer')

    def _video_neural_net_architecture(self, video_frame_shape):

        model = tf.keras.Sequential([
            layers.Input(shape=video_frame_shape, name='video_input'),
            layers.Conv3D(
                filters=96, 
                kernel_size=(1, 11, 11), 
                strides=(1, 4, 4), 
                activation='relu', 
                padding='valid',
                name="video_conv_1"),
            layers.MaxPooling3D(
                pool_size=(1, 3, 3), 
                strides=(1, 2, 2), 
                padding='valid',
                name="video_pool_1"),
            layers.Conv3D(
                filters=256, 
                kernel_size=(1, 5, 5), 
                strides=(1, 1, 1), 
                activation='relu', 
                padding='same',
                name="video_conv_2"),
            layers.MaxPooling3D(
                pool_size=(1, 3, 3), 
                strides=(1, 2, 2), 
                padding='valid',
                name="video_pool_2"),
            layers.Conv3D(filters=384, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name="video_conv_3"),
            layers.Conv3D(filters=384, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name="video_conv_4"),
            layers.Conv3D(filters=256, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same', name="video_conv_5"),
            layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='valid', name="video_pool_3"),
            layers.Flatten(),
            layers.Dense(units=4096, activation='relu', name="video_dense_1"),
            layers.Dropout(0.5, name="video_dropout_1"),
            layers.Dense(units=4096, activation='relu', name="video_dense_2"),
            layers.Dropout(0.5, name="video_dropout_2")
        ],
        name="video_architecture_nn"
        )

        return model
    
    def _audio_neural_net_architecture(self, audio_frame_shape):

        model = tf.keras.Sequential([
            layers.Input(
                shape=audio_frame_shape, 
                name='audio_input'),
            layers.Conv2D(
                filters=24, 
                kernel_size=(3, 3), 
                strides=(1, 1), 
                activation='relu', 
                padding='valid',
                name="audio_conv_1"),
            layers.MaxPooling2D(
                pool_size=(3, 3), 
                strides=(2, 2), 
                padding='valid',
                name="audio_pool_1"),
            layers.Conv2D(
                filters=64, 
                kernel_size=(5, 5), 
                strides=(1, 1), 
                activation='relu', 
                padding='same',
                name="audio_conv_2"),
            layers.MaxPooling2D(
                pool_size=(3, 3), 
                strides=(2, 2), 
                padding='valid',
                name="audio_pool_2"),
            layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name="audio_conv_3"),
            layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name="audio_conv_4"),
            layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name="audio_conv_5"),
            layers.Flatten(),
            layers.Dense(units=1024, activation='relu', name="audio_dense_1"),
            layers.Dropout(0.5, name="audio_dropout_1"),
            layers.Dense(units=1024, activation='relu', name="audio_dense_2"),
            layers.Dropout(0.5, name="audio_dropout_2")
        ],
        name="audio_architecture_nn"
        )

        return model

    def call(self, inputs):

        video_input, audio_input = inputs
        video_output = self.videoNN(video_input)
        audio_output = self.audioNN(audio_input)

        merged = layers.Concatenate()([video_output, audio_output])

        output = self.final_layer(merged)

        return output