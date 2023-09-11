import pypianoroll
import os
import numpy as np


artist = "class 1"

pth = os.path.join("MIDI Music",artist,"midi")
np_path = os.path.join("MIDI Music",artist,"numpy")
files = os.listdir(pth)

for f in files:
    multitrack = pypianoroll.read(os.path.join(pth,f))
    track = np.array(multitrack.tracks)
    if track.shape[0] == 2:
        np.save(os.path.join(np_path,f[:-4]),np.transpose(np.array(multitrack.tracks)[0,:,:] + np.array(multitrack.tracks)[1,:,:]))
    elif track.shape[0] == 1:
        np.save(os.path.join(np_path,f[:-4]),np.transpose(np.array(multitrack.tracks)[0,:,:]))
    else:
        print("More than 2 channels")