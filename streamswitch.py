import os, sys
import time
import random
import json
import subprocess
from pathlib import Path
import numpy as np
import tensorflow as tf
# from tensorflow.keras.utils import plot_model
# from tensorflow.python.platform import build_info as tf_build_info
from data_generation import AdFrameGenerator
from data_generation_parallel import ParallelFrameGenerator
from ad_detector_nn import AdDetectorNN

import pdb

# subprocess.run("nvcc --version", shell=True)
# subprocess.run("nvidia-smi", shell=True)
# print(f'CuDNN VERSION {tf_build_info.build_info["cudnn_version"]}')
# print(f'CUDA VERSION {tf_build_info.build_info["cuda_version"]}')
               
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
# tf.debugging.set_log_device_placement(True)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

physical_devices = tf.config.list_physical_devices("GPU")
print(f'NUM GPUs AVAILABLE: {len(physical_devices)}')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)

# base_path = Path(os.getcwd())
# training_data = base_path.joinpath("training_data")
training_data_root_dir = Path(os.environ["SM_CHANNEL_TRAIN"])
output_dir = Path(os.environ["SM_OUTPUT_DIR"])
# checkpoint_dir = Path(os.environ["SM_CHECKPOINT_DIR"])

# Configuration
NUM_CLASSES = 3
SEQUENCE_LENGTH = 60
BATCH_SIZE = 32
IMG_SIZE = (227, 227, 3) # Height, Width, Channels
AUDIO_FRAME_SHAPE = (104, 26, 2) # Frames, Cepstral Coefficients, Channels
SAMPLES_PER_VIDEO = 64 

TRAINING_DATA_PROPORTION = 0.8
VALIDATION_DATA_PROPORTION = 0.2

# VIDEO_PATHS = [str(training_data.joinpath("0035e03a-2365-4bc7-920e-630050a93e2e").absolute()), str(training_data.joinpath("023d93ac-cbe0-47aa-a91c-06b3d8889e2c").absolute())]
# VIDEO_PATHS = [training_data.joinpath("0035e03a-2365-4bc7-920e-630050a93e2e").absolute(), training_data.joinpath("023d93ac-cbe0-47aa-a91c-06b3d8889e2c").absolute()]
VIDEO_PATHS = [str(d) for d in training_data_root_dir.joinpath("streamswitch_fsx").iterdir()]
random.shuffle(VIDEO_PATHS)
SAMPLES = list(np.repeat(VIDEO_PATHS, SAMPLES_PER_VIDEO))
# random.shuffle(SAMPLES)

TRAINING_SAMPLE_SIZE = int(TRAINING_DATA_PROPORTION * len(SAMPLES))

video_frame_counts = []
audio_frame_counts = []
for path in VIDEO_PATHS:
    with open(Path(path).joinpath("metadata.json"), "r") as f:
        metadata = json.load(f)
        video_frame_counts.append(metadata["total_video_frames"])
        audio_frame_counts.append(metadata["total_audio_frames_16khz"])

labels_data = []
max_video_frames = max(video_frame_counts)
for path in VIDEO_PATHS:   
    label_array = np.load(Path(path).joinpath("labels.npy"))
    # filler_size = max_video_frames - label_array.shape[0]
    # filler_array = np.repeat(3, filler_size).astype(np.uint8)
    # new_array = np.concat([label_array, filler_array])
    labels_data.append(label_array)

video_frame_count_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant(VIDEO_PATHS), tf.constant(video_frame_counts)),
    default_value=-1)

audio_frame_count_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant(VIDEO_PATHS), tf.constant(audio_frame_counts)),
    default_value=-1)

# label_array_table = tf.lookup.StaticHashTable(
#     tf.lookup.KeyValueTensorInitializer(tf.constant(VIDEO_PATHS), tf.constant(np.stack(labels_data), dtype=tf.int32)),
#     default_value=tf.constant([-1] * max_video_frames , dtype=tf.int32)
# )

# 2. Map keys to integer indices
label_array_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(VIDEO_PATHS, tf.range(len(VIDEO_PATHS))),
    default_value=-1
)

# 3. Store actual values in a RaggedTensor
ragged_labels_data = tf.ragged.constant(labels_data)

# 4. Lookup by index
# indices = label_array_table.lookup(tf.constant(VIDEO_PATHS[0]))
# result = tf.gather(ragged_values, indices)

frame_generator = ParallelFrameGenerator(
    label_array_table=label_array_table,
    ragged_labels_data=ragged_labels_data,
    video_frame_count_table=video_frame_count_table,
    audio_frame_count_table=audio_frame_count_table,
    sequence_length=SEQUENCE_LENGTH, 
    target_shape=IMG_SIZE[:2])

training_samples = SAMPLES[:TRAINING_SAMPLE_SIZE]
validation_samples = SAMPLES[TRAINING_SAMPLE_SIZE:]

print(f"NUM TRAINING VIDEO FILES {int(len(VIDEO_PATHS) * TRAINING_DATA_PROPORTION)}")
print(f"NUM VALIDATION VIDEO FILES {int(len(VIDEO_PATHS) * VALIDATION_DATA_PROPORTION)}")
print(f"NUM TRAINING SAMPLES: {len(training_samples)} - STEPS PER EPOCH: {int(len(training_samples) // BATCH_SIZE)}")
print(f"NUM VALIDATION SAMPLES: {len(validation_samples)}")

random.shuffle(training_samples)
train_ds = tf.data.Dataset.from_tensor_slices(training_samples)
# train_ds = dataset.take(TRAINING_SAMPLE_SIZE)
# val_ds = dataset.skip(TRAINING_SAMPLE_SIZE)
train_ds = train_ds.map(frame_generator.get_frame, num_parallel_calls=tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(validation_samples)
val_ds = val_ds.map(frame_generator.get_frame, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

################################################################################

# # Instantiate the generator
# frame_generator = AdFrameGenerator(
#     video_paths=VIDEO_PATHS,
#     samples_per_video=SAMPLES_PER_VIDEO,
#     sequence_length=SEQUENCE_LENGTH,
#     target_shape=IMG_SIZE[:2])
# 
# # Create the dataset
# output_signature = (
#     (
#         tf.TensorSpec(shape=(SEQUENCE_LENGTH, *IMG_SIZE), dtype=tf.float32),
#         tf.TensorSpec(shape=(2 * AdFrameGenerator.NUM_CEPSTRAL_COEFFS, AdFrameGenerator.NUM_CEPSTRAL_FRAMES, 1), dtype=tf.float32)
#     ),
#     tf.TensorSpec(shape=(), dtype=tf.uint8)
# )
# 
# dataset = tf.data.Dataset.from_generator(
#     frame_generator,
#     output_signature=output_signature
# )

################################################################################

model = AdDetectorNN(SEQUENCE_LENGTH, IMG_SIZE, AUDIO_FRAME_SHAPE, NUM_CLASSES)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=output_dir.joinpath("tensorboard"),
    update_freq='batch', 
    write_graph=True)

# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath=checkpoint_dir.joinpath('weights-{epoch:02d}.ckpt'),
#     save_weights_only=True,
#     monitor='val_loss',
#     save_best_only=True
# )

model.fit(train_ds, validation_data=val_ds, epochs=2, steps_per_epoch=TRAINING_SAMPLE_SIZE // BATCH_SIZE, callbacks=[tensorboard_callback])
# model.build(((None, *VIDEO_FRAME_SHAPE), (None, *AUDIO_FRAME_SHAPE)))
# plot_model(model, to_file=base_path.joinpath("ad_detector.png"), show_shapes=True, show_layer_names=True)

# for data, label in train_ds:
#     print("GENERATING DATA!")
#     video_segment, audio_segment = data
#     pdb.set_trace()
#     # result = model.predict((video_grp, audio_grp))