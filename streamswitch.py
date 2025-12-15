from tensorflow.keras import layers, Model

frame_group = 4
num_classes = 2

input_tensor = layers.Input(shape=(frame_group, 227, 227, 3)) # AlexNet typically uses 227x227x3 images

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

output_tensor = layers.Dense(units=num_classes, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output_tensor)