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

physical_devices = tf.config.list_physical_devices("GPU")
print(f'NUM GPUs AVAILABLE: {len(physical_devices)}')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)

base_path = Path(os.getcwd())
training_data = base_path.joinpath("training_data")
# training_data_root_dir = Path(os.environ["SM_CHANNEL_TRAIN"])
# output_dir = Path(os.environ["SM_OUTPUT_DIR"])

# Configuration
NUM_CLASSES = 3
SEQUENCE_LENGTH = 10
BATCH_SIZE = 16
IMG_SIZE = (227, 227, 3) # Height, Width, Channels
AUDIO_FRAME_SHAPE = (AdFrameGenerator.NUM_CEPSTRAL_COEFFS * 2, AdFrameGenerator.NUM_CEPSTRAL_FRAMES, 1)
SAMPLES_PER_VIDEO = 100 

VIDEO_PATHS = [str(training_data.joinpath("0035e03a-2365-4bc7-920e-630050a93e2e").absolute()), str(training_data.joinpath("023d93ac-cbe0-47aa-a91c-06b3d8889e2c").absolute())]
# VIDEO_PATHS = [d for d in training_data_root_dir.joinpath("streamswitch_fsx").iterdir()]
random.shuffle(VIDEO_PATHS)
SAMPLES = list(np.repeat(VIDEO_PATHS, SAMPLES_PER_VIDEO))
print(f"NUM VIDEO FILES {len(VIDEO_PATHS)}")
print(f"TOTAL SAMPLES: {len(SAMPLES)} - STEPS PER EPOCH: {len(SAMPLES) // BATCH_SIZE}")

video_frame_counts = []
audio_frame_counts = []
for path in VIDEO_PATHS:
    with open(Path(path).joinpath("metadata.json"), "r") as f:
        metadata = json.load(f)
        video_frame_counts.append(metadata["total_video_frames"])
        audio_frame_counts.append(metadata["total_audio_frames"])

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

ds = tf.data.Dataset.from_tensor_slices(SAMPLES)
dataset = ds.map(frame_generator.get_frame, num_parallel_calls=tf.data.AUTOTUNE)

dataset = dataset.batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
# model = AdDetectorNN(SEQUENCE_LENGTH, IMG_SIZE, AUDIO_FRAME_SHAPE, NUM_CLASSES)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=output_dir.joinpath("tensorboard"), update_freq='batch', write_graph=True)

# model.fit(dataset, epochs=2, steps_per_epoch=TOTAL_SAMPLES // BATCH_SIZE) # callbacks=[tensorboard_callback])
# model.build(((None, *VIDEO_FRAME_SHAPE), (None, *AUDIO_FRAME_SHAPE)))
# plot_model(model, to_file=base_path.joinpath("ad_detector.png"), show_shapes=True, show_layer_names=True)

for data, label in dataset:
    video_grp, audio_grp = data
    pdb.set_trace()
    # result = model.predict((video_grp, audio_grp))
    print("GENERATING DATA!")