import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from data_generation import AdFrameGenerator
from ad_detector_nn import AdDetectorNN

import pdb

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

base_path = Path(os.getcwd())
training_data = base_path.joinpath("training_data")

# Configuration
NUM_CLASSES = 3
SEQUENCE_LENGTH = 15 
BATCH_SIZE = 8 
IMG_SIZE = (227, 227, 3) # Height, Width, Channels
AUDIO_FRAME_SHAPE = (AdFrameGenerator.NUM_CEPSTRAL_COEFFS * 2, AdFrameGenerator.NUM_CEPSTRAL_FRAMES, 1)
VIDEO_PATHS = [training_data.joinpath("0035e03a-2365-4bc7-920e-630050a93e2e"), training_data.joinpath("023d93ac-cbe0-47aa-a91c-06b3d8889e2c")]
SAMPLES_PER_VIDEO = 100
TOTAL_SAMPLES = len(VIDEO_PATHS) * SAMPLES_PER_VIDEO

# Instantiate the generator
frame_generator = AdFrameGenerator(
    video_paths=VIDEO_PATHS, 
    samples_per_video=SAMPLES_PER_VIDEO,
    sequence_length=SEQUENCE_LENGTH, 
    target_shape=IMG_SIZE[:2])

# Create the dataset
output_signature = (
    (
        tf.TensorSpec(shape=(SEQUENCE_LENGTH, *IMG_SIZE), dtype=tf.float32),
        tf.TensorSpec(shape=(2 * AdFrameGenerator.NUM_CEPSTRAL_COEFFS, AdFrameGenerator.NUM_CEPSTRAL_FRAMES, 1), dtype=tf.float32)
    ),
    tf.TensorSpec(shape=(), dtype=tf.uint8)
)

dataset = tf.data.Dataset.from_generator(
    frame_generator,
    output_signature=output_signature
)

dataset = dataset.batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
model = AdDetectorNN(SEQUENCE_LENGTH, IMG_SIZE, AUDIO_FRAME_SHAPE, NUM_CLASSES)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=base_path.joinpath("logs"), histogram_freq=1, write_graph=True)

model.fit(dataset, epochs=2, steps_per_epoch=TOTAL_SAMPLES // BATCH_SIZE)
# model.build(((None, *VIDEO_FRAME_SHAPE), (None, *AUDIO_FRAME_SHAPE)))
# plot_model(model, to_file=base_path.joinpath("ad_detector.png"), show_shapes=True, show_layer_names=True)

# for data, label in dataset:
#     video_grp, audio_grp = data
#     # result = model.predict((video_grp, audio_grp))
#     pdb.set_trace()
#     print("GENERATING DATA!")