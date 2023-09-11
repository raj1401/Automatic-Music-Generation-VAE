import numpy as np


def concat_notes(array, group_size):
    num_groups = array.shape[0]//group_size
    return array.reshape(num_groups,group_size,128,1)


def expand_groups(array, group_size):
    return array.reshape(array.shape[0]*array.shape[1],128,1)


def trajectories(vae, input_music_list, group_size):
    output_trajectories = []
    for music in input_music_list:
        music_arr = np.transpose(np.load(music))
        music_arr = music_arr.reshape(-1,128,1)
        music_arr = concat_notes(music_arr,group_size)
        print(music_arr.shape)
        # music_arr = music_arr.reshape(-1,128,1)
        output_trajectories.append(vae.encoder.predict(music_arr,verbose=0)[2])
    return output_trajectories


def variable_trajectory(trajectory_list,variation,length, latent_dim):
    num_trajectories = len(trajectory_list)
    trajectory_used = np.random.randint(num_trajectories)
    trajectory_used = 4
    traj = trajectory_list[trajectory_used]
    variation_array = np.random.uniform(-variation,variation,size=(length,latent_dim))
    random_seed = np.random.randint(traj.shape[0]-1)
    random_seed = 0
    new_traj = np.copy(traj[random_seed:random_seed+length,:])
    new_traj = new_traj #+ variation_array
    return trajectory_used, new_traj
