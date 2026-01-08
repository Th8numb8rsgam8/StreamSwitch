import os
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from pydub import AudioSegment
from pathlib import Path

import pdb

NUM_CLASSES = 2

base_path = Path(os.getcwd())

# image_frames = base_path.joinpath("image_frames")
# image_frames.mkdir(exist_ok=True)
# 
# cap = cv2.VideoCapture(base_path.joinpath("Football_Clip.mp4"))
# 
# if not cap.isOpened():
    # print(f"Could not open {video_path}")
    # sys.exit(1)
# 
# frame_num = 0
# while True:
# 
    # ret, frame = cap.read()
# 
    # if not ret:
        # break
# 
    # jpeg_file = image_frames.joinpath(image_frames, f"frame_{frame_num}.jpg")
    # cv2.imwrite(jpeg_file, frame)
# 
    # if frame_num % 100 == 0: 
        # print(f"Frame {frame_num} saved.")
    # 
    # frame_num += 1
# 
# cap.release()

################################################################################

# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = cap.get(cv2.CAP_PROP_FPS)
# 
# labels = np.random.randint(0, 1, size=total_frames)

# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
# 
# ret, frame = cap.read()

# cv2.imshow(f"Frame {frame_number}", frame)
# cv2.waitKey(0)

class FrameGenerator:

    def __init__(self, video_paths, group_size, target_shape=(224, 224)):

        self.video_paths = video_paths
        self.group_size = group_size
        self.target_shape = target_shape

    # def get_files_and_labels(self):
    #     video_files = []
    #     for class_index, class_name in enumerate(self.class_names):
    #         class_path = os.path.join(self.data_path, class_name)
    #         for video_name in os.listdir(class_path):
    #             video_files.append((os.path.join(class_path, video_name), class_index))
    #     if self.training:
    #         np.random.shuffle(video_files)
    #     return video_files

    def frames_from_video_file(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Could not open {video_path}")
            return
            # sys.exit(1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
 
        num_groups, unused_frames = divmod(total_frames, FRAME_GROUP)
 
        for _ in range(num_groups):
            frame_grp = []
            for _ in range(self.group_size):
                ret, frame = cap.read()
                frame = tf.image.convert_image_dtype(frame, tf.float32)
                frame = tf.image.resize(frame, self.target_shape)
                frame_grp.append(frame)
            yield tf.stack(frame_grp, axis=0)
     
        cap.release()
           
    def __call__(self):
        for video_path in self.video_paths:
            for frame_grp in self.frames_from_video_file(video_path):
                yield frame_grp # tf.keras.utils.to_categorical(label, num_classes=len(self.class_names))
 
# Configuration
FRAME_GROUP = 4 
IMG_SIZE = (112, 112) # Height, Width
VIDEO_PATHS = [base_path.joinpath("Family_Guy.mp4"), base_path.joinpath("Football_Clip.mp4")]

audio = AudioSegment.from_file(VIDEO_PATHS[0], format="mp4")
num_channels = audio.channels
result = audio.get_sample_slice(0, 20).get_array_of_samples()
result_array = np.array(result).reshape(-1, num_channels)
pdb.set_trace()

# Instantiate the generator
frame_generator = FrameGenerator(
    video_paths=VIDEO_PATHS, 
    group_size=FRAME_GROUP, 
    target_shape=IMG_SIZE)

# Create the dataset
output_signature = tf.TensorSpec(shape=(FRAME_GROUP, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)

dataset = tf.data.Dataset.from_generator(
    frame_generator,
    output_signature=output_signature # or use output_signature
)

dataset = dataset.prefetch(tf.data.AUTOTUNE)

for video_grp in dataset:
    pdb.set_trace()

################################################################################

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()

# def process_with_cv2(path):
# 
#     cap = cv2.VideoCapture(path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
# 
#     num_groups, unused_frames = divmod(total_frames, FRAME_GROUP)
# 
#     if not cap.isOpened():
#         print(f"Could not open {video_path}")
#         sys.exit(1)
# 
#     frames = []
#     for _ in range(total_frames - unused_frames):
#         ret, frame = cap.read()
#         frame = tf.image.convert_image_dtype(frame, tf.float32)
#         frame = tf.image.resize(frame, (64, 64))
#         frames.append(frame)
#     
#     cap.release()
# 
#     return tf.stack(frames, axis=0)
# 
# def process_path(path):
# 
#     video_frames = tf.numpy_function(
#         func=process_with_cv2,
#         inp=[path],
#         Tout=[tf.float32]
#     )
# 
#     # num_frame_grps = int(num_frames.numpy() / FRAME_GROUP)
#     # frame_groups = tf.split(video_frames, num_frame_grps, axis=0)
#     # labels = [f"label{i}" for i in range(num_frame_grps)]
# 
#     # return video_frames, "labels"
#     return video_frames

# def split_tensor(tensor, frame_grp):
# 
#     # shape = tf.shape(tensor)
# 
#     # frame_groups = []
#     # for i in range(shape[0]):
#     #     frame_group = []
#     #     for j in range(n):
#     #         frame_group.append(tensor[i*n+j,:,:,:])
#     #     frame_groups.append(tf.stack(frame_group, axis=0))
#     # num_groups = tf.math.divide(shape[0], tf.constant(n))
# 
#     # num_groups = int(shape[0] / n)
#     # tf.print(f"NUM GROUPS: {num_groups}")
#     frame_groups = tf.split(tensor, num_or_size_splits=frame_grp, axis=0)
# 
#     # new_shape = (n, shape[1], shape[2], shape[3])
#     # reshaped_tensor = tf.reshape(tensor, new_shape)
# 
#     return tf.data.Dataset.from_tensor_slices(frame_groups)

# train_ds = tf.data.Dataset.from_tensor_slices(paths)
# train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
# train_ds = train_ds.batch(FRAME_GROUP, drop_remainder=True)
# train_ds = train_ds.flat_map(lambda x: split_tensor(x, tf.shape(x)[0] * [FRAME_GROUP])) # tf.math.divide(tf.shape(x)[0], tf.constant(FRAME_GROUP))))

# train_ds = train_ds.shuffle(bffer_size=len(train_ds)).batch(64, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

# for video in train_ds:
#     pdb.set_trace()
#     print(f"Image shape: {image.numpy().shape}")
#     print(f"Labels: {labels}")


input_tensor = layers.Input(shape=(FRAME_GROUP, 227, 227, 3)) # AlexNet typically uses 227x227x3 images

x = layers.Conv3D(filters=96, kernel_size=(1, 11, 11), strides=(1, 4, 4), activation='relu', padding='valid')(input_tensor)
x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='valid')(x)

x = layers.Conv3D(filters=256, kernel_size=(1, 5, 5), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='valid')(x)

x = layers.Conv3D(filters=384, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = layers.Conv3D(filters=384, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = layers.Conv3D(filters=256, kernel_size=(1, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
x = layers.MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding='valid')(x)

x = layers.Flatten()(x)
x = layers.Dense(units=4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(units=4096, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output_tensor = layers.Dense(units=NUM_CLASSES, activation='sigmoid')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)