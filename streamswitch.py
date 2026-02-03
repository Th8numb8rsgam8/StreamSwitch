import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from data_generation import AdFrameGenerator
from ad_detector_nn import AdDetectorNN

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

base_path = Path(os.getcwd())

# Configuration
NUM_CLASSES = 1
FRAME_GROUP = 6 
IMG_SIZE = (227, 227) # Height, Width
VIDEO_FRAME_SHAPE = (FRAME_GROUP, *IMG_SIZE, 3)
AUDIO_FRAME_SHAPE = (AdFrameGenerator.NUM_CEPSTRAL_COEFFS * 2, AdFrameGenerator.NUM_CEPSTRAL_FRAMES, 1)
VIDEO_PATHS = [base_path.joinpath("c4814504-c3d3-4e8c-a896-ea9175e4a2cf.mp4")]

# Instantiate the generator
frame_generator = AdFrameGenerator(
    video_paths=VIDEO_PATHS, 
    video_group_size=FRAME_GROUP, 
    target_shape=IMG_SIZE)

# Create the dataset
output_signature = (
    (
        tf.TensorSpec(shape=(None, FRAME_GROUP, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 2 * AdFrameGenerator.NUM_CEPSTRAL_COEFFS, AdFrameGenerator.NUM_CEPSTRAL_FRAMES, 1), dtype=tf.float32)
    ),
    tf.TensorSpec(shape=(None,), dtype=tf.int64)
)

dataset = tf.data.Dataset.from_generator(
    frame_generator,
    output_signature=output_signature # or use output_signature
)

dataset = dataset.prefetch(tf.data.AUTOTUNE)

model = AdDetectorNN(VIDEO_FRAME_SHAPE, AUDIO_FRAME_SHAPE, NUM_CLASSES)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=base_path.joinpath("logs"), histogram_freq=1, write_graph=True)

# model.fit(dataset, epochs=1)
# model.build(((None, *VIDEO_FRAME_SHAPE), (None, *AUDIO_FRAME_SHAPE)))
# plot_model(model, to_file=base_path.joinpath("ad_detector.png"), show_shapes=True, show_layer_names=True)

for data, label in dataset:
    video_grp, audio_grp = data
    # result = model.predict((video_grp, audio_grp))
    pdb.set_trace()
    print("GENERATING DATA!")