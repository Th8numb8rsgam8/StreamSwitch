import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Model, utils, applications

import pdb

class AdDetectorNN(Model):

    def __init__(self, sequence_length, video_frame_shape, video_frames_per_audio, num_classes):

        super(AdDetectorNN, self).__init__()
        # self.videoNN = self._video_neural_net_architecture(sequence_length, video_frame_shape)
        # self.audioNN = self._audio_neural_net_architecture(audio_frame_shape)
        # self.final_layer = layers.Dense(units=num_classes, activation='softmax', name='final_layer', dtype='float32')

        self._SEQ_LEN = sequence_length
        self._FRAMES_PER_STEP = video_frames_per_audio
        self._VIDEO_FRAME_SHAPE = video_frame_shape
        self._NUM_CLASSES = num_classes
        self._VIDEO_INPUT_SHAPE = (self._SEQ_LEN * self._FRAMES_PER_STEP, *self._VIDEO_FRAME_SHAPE)
        self._AUDIO_INPUT_SHAPE = (7680 * (self._SEQ_LEN + 1),)

        self._yamnet_location = 'https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1'
        self.NeuralNet = self._edge_device_ad_detector()

        # self.videoNN = self._video_neural_net_architecture(sequence_length, video_frame_shape)
        # self.audioNN = self._audio_neural_net_architecture(audio_frame_shape)
        # self.final_layer = layers.Dense(units=num_classes, activation='softmax', name='final_layer')

    def _edge_device_ad_detector(self):

        video_input = tf.keras.Input(shape=self._VIDEO_INPUT_SHAPE, dtype=tf.float32, name="video_frame_sequence")
        audio_input = tf.keras.Input(shape=self._AUDIO_INPUT_SHAPE, dtype=tf.float32, name="raw_audio")

        # mobilenet = applications.MobileNetV3Small(
        #     input_shape=self._VIDEO_FRAME_SHAPE, 
        #     include_top=False, 
        #     pooling='avg', 
        #     weights='imagenet'
        # )
        # mobilenet.trainable = False
        
        # time_distributed = layers.TimeDistributed(mobilenet, name="Time_Distributed_MobileNet")(video_input)
        mobilenet = TimeDistributedMobileNet(self._VIDEO_FRAME_SHAPE)(video_input)
        reshaped_video_features = layers.Reshape((self._SEQ_LEN, self._FRAMES_PER_STEP, 576), name="reshape_video_frames")(mobilenet)
        visual_context = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=2), name="intra_step_pooling")(reshaped_video_features)

        # yamnet = hub.KerasLayer(self._yamnet_location, trainable=False, name="yamnet_layer")
        yamnet = YamnetEmbeddingExtractor(self._yamnet_location, name="yamnet_layer")
        audio_embeddings = yamnet(audio_input)

        fused_sequence = layers.Concatenate(axis=-1, name="video_audio_fusion")([audio_embeddings, visual_context])
        x = layers.GRU(units=128, return_sequences=False, name="trainable_gru")(fused_sequence)
        x = layers.Dense(48, activation='relu')(x)
        predictions = tf.keras.layers.Dense(units=self._NUM_CLASSES, activation='softmax', dtype=tf.float32, name="output")(x)
 
        model = tf.keras.Model(inputs=[audio_input, video_input], outputs=predictions)

        return model


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

        # video_input, audio_input = inputs
        # video_output = self.videoNN(video_input)
        # audio_output = self.audioNN(audio_input)
        # merged = layers.Concatenate()([video_output, audio_output])
        # output = self.final_layer(merged)

        output = self.NeuralNet(inputs)

        return output

class TimeDistributedMobileNet(layers.Layer):
    def __init__(self, video_frame_shape, **kwargs):

        # Passing dtype='float32' blocks the mixed_float16 policy from infecting MobileNet
        super(TimeDistributedMobileNet, self).__init__(dtype='float32', **kwargs)
        
        self.video_frame_shape = video_frame_shape
        # Load MobileNetV3 Small with ImageNet weights (Requires exactly 3 channels)
        self.mobilenet = applications.MobileNetV3Small(
            input_shape=self.video_frame_shape, 
            include_top=False, 
            pooling='avg', 
            weights='imagenet',
            include_preprocessing=True
        )
        self.mobilenet.trainable = False

    def call(self, video_inputs):

        # Expected input shape: (Batch_Size, TimeSteps, 224, 224, 3)
        input_shape = tf.shape(video_inputs)
        batch_size = input_shape[0]
        timesteps = input_shape[1]
        
        # Collapse Batch and Time dimensions to mimic TimeDistributed manually
        # Reshaped shape: (Batch_Size * TimeSteps, 224, 224, 3)
        flattened_frames = tf.reshape(video_inputs, [-1, *self.video_frame_shape])
        flattened_frames = tf.cast(flattened_frames, tf.float32)
        
        # Process frames purely in safe float32 space
        features = self.mobilenet(flattened_frames) # Shape: (Batch_Size * TimeSteps, 576)
        
        # Re-inflate the Time axis for your sequential GRU layer
        # Restored shape: (Batch_Size, TimeSteps, 576)
        video_sequence = tf.reshape(features, [batch_size, timesteps, 576])

        return video_sequence

class YamnetEmbeddingExtractor(layers.Layer):

    def __init__(self, hub_url, name=None, **kwargs):
        super(YamnetEmbeddingExtractor, self).__init__(name=name, dtype='float32', **kwargs)
        self.hub_url = hub_url
        self.yamnet = hub.KerasLayer(hub_url, trainable=False, name=name)

    def call(self, inputs):

        def extract_single_waveform(waveform):
            waveform = tf.cast(waveform, tf.float32)
            _, embeddings, _ = self.yamnet(waveform)
            return embeddings

        return tf.map_fn(
            fn=extract_single_waveform,
            elems=inputs,
            fn_output_signature=tf.float32
        )

    def compute_output_shape(self, input_shape):

        return (input_shape[0], None, 1024)

    def get_config(self):

        config = super(YamnetEmbeddingExtractor, self).get_config()
        config.update({"hub_url": self.hub_url})
        return config

# import tensorflow as tf
# import tensorflow_hub as hub
# 
# # 1. ARCHITECTURE HYPERPARAMETERS
# SEQ_LEN = 10                     # The GRU looks at a history of 10 temporal steps
# FRAMES_PER_STEP = 3              # 3 video frames per temporal step
# AUDIO_SAMPLES_PER_STEP = 15360   # ~0.96s window at 16kHz (YAMNet standard segment size)
# 
# # Calculate total expected raw inputs for the entire sequence window
# TOTAL_AUDIO_SAMPLES = AUDIO_SAMPLES_PER_STEP * SEQ_LEN
# AUDIO_INPUT_SHAPE = (TOTAL_AUDIO_SAMPLES,)
# VIDEO_INPUT_SHAPE = (SEQ_LEN, FRAMES_PER_STEP, 224, 224, 3)
# 
# YAMNET_HANDLE = 'https://tfhub.dev'
# 
# # 2. DEFINE THE MULTI-INPUT FUNCTIONAL GRAPH
# def build_end_to_end_multimodal_model():
#     # --- INPUT LAYER DEFINITIONS ---
#     audio_input = tf.keras.Input(shape=AUDIO_INPUT_SHAPE, dtype=tf.float32, name="raw_audio_waveform")
#     video_input = tf.keras.Input(shape=VIDEO_INPUT_SHAPE, dtype=tf.float32, name="video_frames_sequence")
# 
#     # ==========================================
#     # BRANCH A: AUDIO PROCESSING (YAMNet Layer)
#     # ==========================================
#     # Initialize YAMNet as a non-trainable Keras Layer
#     yamnet_layer = hub.KerasLayer(YAMNET_HANDLE, trainable=False, name="frozen_yamnet")
#     
#     # YAMNet automatically slices long 1D audio inputs into 0.96s segments internally
#     # Returns shape: (SEQ_LEN, 1024)
#     audio_embeddings = yamnet_layer(audio_input)
# 
#     # ==========================================
#     # BRANCH B: VIDEO PROCESSING (MobileNet Layer)
#     # ==========================================
#     # Initialize MobileNetV3Small as a non-trainable layer wrapper
#     mobilenet_backbone = tf.keras.applications.MobileNetV3Small(
#         input_shape=(224, 224, 3), 
#         include_top=False, 
#         pooling='avg', 
#         weights='imagenet'
#     )
#     mobilenet_backbone.trainable = False  # Explicitly freeze the visual backbone
#     
#     # Because video_input is a 5D tensor (Batch, SEQ_LEN, FRAMES_PER_STEP, H, W, C),
#     # we use TimeDistributed to apply MobileNet to every single frame across time.
#     # TimeDistributed flattens the dimensions to process images, then restores the temporal shape.
#     frame_features = tf.keras.layers.TimeDistributed(
#         tf.keras.layers.TimeDistributed(mobilenet_backbone, name="frozen_mobilenet"),
#         name="temporal_frame_processor"
#     )(video_input) # Output shape: (Batch, SEQ_LEN, FRAMES_PER_STEP, 576)
# 
#     # Compress the 3 frames within each step down to 1 visual vector using average pooling
#     # Reduces shape from (Batch, SEQ_LEN, 3, 576) down to (Batch, SEQ_LEN, 576)
#     visual_context = tf.keras.layers.Lambda(
#         lambda x: tf.reduce_mean(x, axis=2), 
#         name="intra_step_video_pooling"
#     )(frame_features)
# 
#     # ==========================================
#     # MULTIMODAL FUSION & TRAINABLE CLASSIFIER
#     # ==========================================
#     # Combine frozen 1024-dim audio features and frozen 576-dim visual features
#     # Output shape: (Batch, SEQ_LEN, 1600)
#     # Note: Keras layer concatenation handles adding the virtual batch dimension automatically
#     fused_sequence = tf.keras.layers.Concatenate(axis=-1, name="late_fusion")([audio_embeddings, visual_context])
# 
#     # Trainable GRU Head - This is what learns your specific 'ad', 'not_ad', 'transition' logic
#     x = tf.keras.layers.GRU(128, return_sequences=False, dropout=0.3, recurrent_dropout=0.0, name="trainable_gru")(fused_sequence)
#     x = tf.keras.layers.Dense(48, activation='relu', name="trainable_dense")(x)
#     predictions = tf.keras.layers.Dense(3, activation='softmax', name="output_classification")(x)
# 
#     # Instantiate the unified Model
#     model = tf.keras.Model(inputs=[audio_input, video_input], outputs=predictions, name="unified_edge_model")
#     
#     # Compile the model
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model
# 
# # 3. INSTANTIATE AND INSPECT THE STRUCTURE
# multimodal_model = build_end_to_end_multimodal_model()
# multimodal_model.summary()

################################################################################

# import tensorflow as tf
# import tensorflow_hub as hub
# 
# # 1. PARAMETERS
# SEQ_LEN = 10                     # 10 sequence steps
# FRAMES_PER_STEP = 4              # 4 frames per step
# TOTAL_IMAGES = SEQ_LEN * FRAMES_PER_STEP  # 40 total frames in the sequence
# 
# AUDIO_INPUT_SHAPE = (7680 * (SEQ_LEN + 1),) # Synchronized raw audio samples
# VIDEO_INPUT_SHAPE = (TOTAL_IMAGES, 224, 224, 3) # Combined 4D Image Tensor
# 
# YAMNET_HANDLE = 'https://tfhub.dev'
# 
# def build_flattened_multimodal_model():
#     # --- INPUT LAYERS ---
#     audio_input = tf.keras.Input(shape=AUDIO_INPUT_SHAPE, dtype=tf.float32, name="raw_audio")
#     video_input = tf.keras.Input(shape=VIDEO_INPUT_SHAPE, dtype=tf.float32, name="combined_video_frames")
# 
#     # ==========================================
#     # BRANCH A: AUDIO PROCESSING (YAMNet)
#     # ==========================================
#     yamnet_layer = hub.KerasLayer(YAMNET_HANDLE, trainable=False, name="frozen_yamnet")
#     audio_embeddings = yamnet_layer(audio_input) # Shape: (Batch, SEQ_LEN, 1024)
# 
#     # ==========================================
#     # BRANCH B: VIDEO PROCESSING (MobileNet)
#     # ==========================================
#     mobilenet_backbone = tf.keras.applications.MobileNetV3Small(
#         input_shape=(224, 224, 3), include_top=False, pooling='avg', weights='imagenet'
#     )
#     mobilenet_backbone.trainable = False
# 
#     # Only ONE TimeDistributed wrapper is needed here to process the 40 flat frames
#     flat_features = tf.keras.layers.TimeDistributed(
#         mobilenet_backbone, name="single_wrapper_mobilenet"
#     )(video_input) # Output shape: (Batch, 40, 576)
# 
#     # Reshape the features back into their distinct temporal groups
#     # Shape changes from (Batch, 40, 576) -> (Batch, 10, 4, 576)
#     reshaped_features = tf.keras.layers.Reshape((SEQ_LEN, FRAMES_PER_STEP, 576), name="restore_time_structure")(flat_features)
# 
#     # Pool across the 4 frames to create 1 visual context vector per step
#     # Shape changes from (Batch, 10, 4, 576) -> (Batch, 10, 576)
#     visual_context = tf.keras.layers.Lambda(
#         lambda x: tf.reduce_mean(x, axis=2), name="intra_step_pooling"
#     )(reshaped_features)
# 
#     # ==========================================
#     # FUSION & TRAINABLE GRU CLASSIFIER
#     # ==========================================
#     fused_sequence = tf.keras.layers.Concatenate(axis=-1, name="fusion")([audio_embeddings, visual_context])
# 
#     x = tf.keras.layers.GRU(128, return_sequences=False, name="trainable_gru")(fused_sequence)
#     x = tf.keras.layers.Dense(48, activation='relu')(x)
#     predictions = tf.keras.layers.Dense(3, activation='softmax', name="output")(x)
# 
#     model = tf.keras.Model(inputs=[audio_input, video_input], outputs=predictions)
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#     return model
# 
# multimodal_model = build_flattened_multimodal_model()
# multimodal_model.summary()