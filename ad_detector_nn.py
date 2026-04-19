import tensorflow as tf
from tensorflow.keras import layers, Model, utils

import pdb

class AdDetectorNN(Model):

    def __init__(self, sequence_length, video_frame_shape, audio_frame_shape, num_classes):

        super(AdDetectorNN, self).__init__()
        self.videoNN = self._video_neural_net_architecture(sequence_length, video_frame_shape)
        self.audioNN = self._audio_neural_net_architecture(audio_frame_shape)
        self.final_layer = layers.Dense(units=num_classes, activation='softmax', name='final_layer')

    def _video_neural_net_architecture(self, sequence_length, video_frame_shape):

        alexnet = self._alexnet_cnn(video_frame_shape)
        sequence_input = layers.Input(shape=(sequence_length, *video_frame_shape), name='video_input')
        time_distributed = layers.TimeDistributed(alexnet, name="Time_Distributed_AlexNet")(sequence_input)
        sequence_output = layers.LSTM(units=64, return_sequences=False)(time_distributed)
        model = Model(inputs=sequence_input, outputs=sequence_output, name="Video_Sequence_Model")

        # utils.plot_model(
        #     model,
        #     show_shapes=True,
        #     show_layer_names=True,
        #     expand_nested=True
        # )

        return model
    
    def _alexnet_cnn(self, video_frame_shape):

        image_input = layers.Input(shape=video_frame_shape, name='image_input')

        # 1st Convolutional Layer
        x = layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', name="video_conv_1")(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name="video_pool_1")(x)

        # 2nd Convolutional Layer
        x = layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same',name="video_conv_2")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name="video_pool_2")(x)

        # 3rd Convolutional Layer
        x = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', name="video_conv_3")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # 4th Convolutional Layer
        x = layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', name="video_conv_4")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        # 5th Convolutional Layer
        x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name="video_conv_5")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name="video_pool_3")(x)

        # Flatten for fully connected layers
        x = layers.Flatten()(x)

        # 1st Fully Connected Layer
        x = layers.Dense(units=4096, activation='relu', name="video_dense_1")(x)
        x = layers.Dropout(0.5, name="video_dropout_1")(x)

        # 2nd Fully Connected Layer
        x = layers.Dense(units=4096, activation='relu', name="video_dense_2")(x)
        outputs = layers.Dropout(0.5, name="video_dropout_2")(x)

        model = Model(inputs=image_input, outputs=outputs, name="alexnet_architecture_nn")

        return model
    
    def _audio_neural_net_architecture(self, audio_frame_shape):

        audio_input = layers.Input(shape=audio_frame_shape, name='audio_input')

        x = layers.Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), padding='valid', name="audio_conv_1")(audio_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name="audio_pool_1")(x)

        x = layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', name="audio_conv_2")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name="audio_pool_2")(x)

        x = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', name="audio_conv_3")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=96, kernel_size=(3, 3), strides=(1, 1), padding='same', name="audio_conv_4")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name="audio_conv_5")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        x = layers.Flatten()(x)

        x = layers.Dense(units=1024, activation='relu', name="audio_dense_1")(x)
        x = layers.Dropout(0.5, name="audio_dropout_1")(x)

        x = layers.Dense(units=1024, activation='relu', name="audio_dense_2")(x)
        outputs = layers.Dropout(0.5, name="audio_dropout_2")(x)

        model = Model(inputs=audio_input, outputs=outputs, name="audio_architecture_nn")

        return model

    def call(self, inputs):

        video_input, audio_input = inputs
        print(f"FEEDING DATA: VIDEO SHAPE - {video_input.shape}; AUDIO SHAPE - {audio_input.shape}")
        video_output = self.videoNN(video_input)
        audio_output = self.audioNN(audio_input)

        merged = layers.Concatenate()([video_output, audio_output])

        output = self.final_layer(merged)

        return output