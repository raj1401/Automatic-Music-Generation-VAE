import os
import numpy as np
from VariationalAutoEncoder import *
from accessory_functions import *


##############################################
############# Hyper-Parameters ###############

latent_dim = 3

hl1 = 128
hl2 = 90
hl3 = 50

group_size = 24

epochs = 5
batch_size = 5

step_size = 0.15  # For random-walk

##############################################
##############################################

artist = "class 1"
pth = os.path.join("MIDI Music",artist)
music_array = np.load(os.path.join(pth,artist+".npy"))
music_array.reshape(-1,128,1)
print(music_array.shape)
music_array = music_array.reshape((-1,128,1))

music_array = concat_notes(music_array, group_size)
print(music_array.shape)

# Creating the Encoder, Decoder, and VAE for Piano for the selected artist
enc = encoder((group_size,128,1),latent_dim,hl1,hl2,hl3)
enc.make_model()
print(enc.my_summary())

dec = decoder((group_size,128,1),latent_dim,hl2,hl1,hl1)
dec.make_model()
print(dec.my_summary())

vae = VAE(enc.model,dec.model)
vae.compile(optimizer=keras.optimizers.Adam())


# Training the VAE
vae.fit(music_array, epochs=epochs, batch_size=batch_size)


# Getting the train_data trajectories in latent space
tra_path = os.path.join(pth,"numpy")
tra_sets = os.listdir(tra_path)
tra_set_paths = []
for s in tra_sets:
    tra_set_paths.append(os.path.join(tra_path,s))
train_trajectories_list = trajectories(vae,tra_set_paths, group_size)

# Making New Music
music_length = 10000

traj_used, new_traj = variable_trajectory(train_trajectories_list,step_size,music_length,latent_dim)
print(f"Generating New Music using {tra_sets[traj_used]} as starter.")
output_music = vae.decoder.predict(new_traj,verbose=0)
print(output_music.shape)
output_music = expand_groups(output_music, group_size)

output_music = np.transpose(output_music).reshape(128,-1)

print(output_music.shape)
pth = os.path.join("MIDI Music",artist)
np.save(os.path.join(pth,f"new_vae_music_{artist}.npy"),output_music)