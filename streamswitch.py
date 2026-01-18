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
 
# Configuration
NUM_CLASSES = 1
FRAME_GROUP = 5 
IMG_SIZE = (227, 227) # Height, Width
VIDEO_FRAME_SHAPE = (FRAME_GROUP, IMG_SIZE[0], IMG_SIZE[1], 3)
AUDIO_FRAME_SHAPE = (AdFrameGenerator.NUM_CEPSTRAL_COEFFS * 2, AdFrameGenerator.NUM_CEPSTRAL_FRAMES, 1)
VIDEO_PATHS = [base_path.joinpath("Family_Guy.mp4")]

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

model.fit(dataset, epochs=1, callbacks=[tensorboard_callback])
# model.build(((None, *VIDEO_FRAME_SHAPE), (None, *AUDIO_FRAME_SHAPE)))
# plot_model(model, to_file=base_path.joinpath("ad_detector.png"), show_shapes=True, show_layer_names=True)

# for data, label in dataset:
#     video_grp, audio_grp = data
#     # result = model.predict((video_grp, audio_grp))
#     pdb.set_trace()
#     print("GENERATING DATA!")