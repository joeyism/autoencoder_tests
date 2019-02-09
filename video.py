import gc

import imageio
import numpy as np
from tqdm import tqdm

import autoencoder


filename = "bladerunner2049.mp4"
reader = imageio.get_reader(filename, "ffmpeg")
fps = reader.get_meta_data()['fps']

frames = [val for val in tqdm(reader, desc="Reading video")]
del reader
gc.collect()
N = len(frames) - 1
print("Video has {} frames".format(N))

h, w, z = frames[0].shape
frames = np.stack(frames)
frames = frames.reshape((-1, h*w*z))

autoencode_and_generate = autoencoder.create_model(h, w, z)
gc.collect()
output_frames = autoencode_and_generate(frames, frames)

import ipdb
ipdb.set_trace()

writer = imageio.get_writer("output/" + filename, fps=fps)
