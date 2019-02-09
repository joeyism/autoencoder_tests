import gc

import imageio
import numpy as np
from tqdm import tqdm

import autoencoder


filename = "bringmethanos.mp4"
reader = imageio.get_reader(filename, "ffmpeg")
fps = reader.get_meta_data()['fps']

frames = [val for val in tqdm(reader, desc="Reading video")]
del reader
gc.collect()
N = len(frames) - 1
print("Video has {} frames".format(N))

h, w, z = frames[0].shape
print((h, w, z))
frames = np.stack(frames)

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

input_img = Input(shape=(h, w, z))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.callbacks import TensorBoard

autoencoder.fit(frames, frames,
                epochs=10,
                batch_size=128,
                validation_data=(frames, frames),
                verbose=1,
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

frames_recon = autoencoder.predict(frames)
del frames
gc.collect()
writer = imageio.get_writer("output/" + filename , fps=fps)
try:
    for frame in tqdm(frames_recon, desc="Writing"):
        writer.append(frame)
finally:
    writer.close()
