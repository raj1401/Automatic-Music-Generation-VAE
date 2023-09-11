import numpy as np
import os


artist = "class 1"

np_path = os.path.join("MIDI Music",artist,"numpy")
files = os.listdir(np_path)
print(np_path)

starter = np.load(os.path.join(np_path,files[0]))
div = np.max(starter,axis=0)
starter = np.divide(starter, div, out=np.zeros(starter.shape,dtype=float), where=(div!=0))
for f in files[1:]:
    new_array = np.load(os.path.join(np_path,f))
    div = np.max(new_array,axis=0)
    new_array = np.divide(new_array, div, out=np.zeros(new_array.shape,dtype=float), where=(div!=0))
    starter = np.concatenate((starter,new_array),axis=1)

save_path = os.path.join("MIDI Music",artist)
np.save(os.path.join(save_path,artist),np.transpose(starter))