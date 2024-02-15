import os 
from glob import glob 
import cv2 
import numpy as np
from itertools import accumulate
import pickle 

'''
dataset_generator.py --> numpy pickle for diffusion policy dataset 

1. collect overhead images  
2. collect agent pos 

create dataset 

3. duplicate and shift to make action array  
4. episode ends array
5. zarr or pickle 
'''

# 1. collect overhead images & 2. collect agent pos 
episode_paths = glob('../rlbench_data/reach_target/*/episodes/episode*')
image_wrist_root = 'wrist_rgb'
image_overhead_root = 'overhead_rgb'
agent_pos_filename = 'low_dim_obs.pkl'
episode_ends = []
images_wrist = []
images_overhead = []
agent_pos = []
joint_pos = [] 

for episode_path in episode_paths:
    images_dir = os.path.join(episode_path, image_wrist_root, '*')
    images_per_episode = np.array(list(map(cv2.imread, sorted(glob(images_dir), key=os.path.getctime))))[:-1] # shifting to the left
    images_wrist.append(images_per_episode)

    images_dir = os.path.join(episode_path, image_overhead_root, '*')
    images_per_episode = np.array(list(map(cv2.imread, sorted(glob(images_dir), key=os.path.getctime))))[:-1] # shifting to the left 
    images_overhead.append(images_per_episode)

    sequence_length = images_per_episode.shape[0]
    episode_ends.append(sequence_length)

    agent_pos_filepath = os.path.join(episode_path, agent_pos_filename)
    with open(agent_pos_filepath, 'rb') as readfile: 
        data = pickle.load(readfile)
    
    # shifting and appending 
    joint_pos_per_episode = np.array(list(map(lambda x: x.joint_positions, data)))[:-1]
    joint_pos.append(joint_pos_per_episode)
    agent_pos_per_episode = np.array(list(map(lambda x: x.gripper_pose, data)))[1:]
    agent_pos.append(agent_pos_per_episode)

episode_ends = np.array(list(accumulate(episode_ends)))

images_wrist = np.concatenate(images_wrist, axis=0, dtype=np.float32)
images_overhead = np.concatenate(images_overhead, axis=0, dtype=np.float32)
agent_pos = np.concatenate(agent_pos, axis=0, dtype=np.float32)
joint_pos = np.concatenate(joint_pos, axis=0, dtype=np.float32)

print(images_overhead.shape, images_overhead.dtype)
print(agent_pos.shape, agent_pos.dtype)
print()
print(agent_pos[0])
print(type(episode_ends), episode_ends)


# 3. duplicate and shift to make action array
# action = agent_pos.copy()

# 일단 shifting 나중에 해보고 학습이랑 검증해보자 
import torch 


def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

class TaskImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                #  dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):

        train_image_data_wrist = np.moveaxis(images_wrist, -1,1)
        train_image_data_overhead = np.moveaxis(images_overhead, -1,1)
        # (N,3,128,128)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'joint_pos': joint_pos,
            'action': agent_pos
        }
        # episode_ends = episode_ends

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()

        key = 'action'
        stats[key] = get_data_stats(train_data['action'][:, :3])
        train_data['action'] = np.concatenate([normalize_data(train_data['action'][:, :3], stats[key]), train_data['action'][:, 3:]], axis=-1)

        # images are already normalized
        # normalized_train_data['image'] = train_image_data
        train_data['image_wrist'] = train_image_data_wrist
        train_data['image_overhead'] = train_image_data_overhead

        self.indices = indices
        self.stats = stats
        # self.normalized_train_data = normalized_train_data
        self.train_data = train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        nsample['image_wrist'] = nsample['image_wrist'][:self.obs_horizon,:]
        nsample['image_overhead'] = nsample['image_overhead'][:self.obs_horizon,:]
        nsample['joint_pos'] = nsample['joint_pos'][:self.obs_horizon,:]
        # nsample['action'] = nsample['action'][:self.obs_horizon,:]
        return nsample


pred_horizon = 16
obs_horizon = 4
action_horizon = 8

dataset = TaskImageDataset(
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)

# save training data statistics (min, max) for each dim
stats = dataset.stats

import pickle 
with open('stats.pkl', 'wb') as wf:
    pickle.dump(stats, wf)






print("dataset ready")
# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

# visualize data in batch
batch = next(iter(dataloader))
print("batch['image_overhead'].shape:", batch['image_overhead'].shape)
print("batch['image_wrist'].shape:", batch['image_wrist'].shape)
print("batch['joint_pos'].shape:", batch['joint_pos'].shape)
print("batch['action'].shape", batch['action'].shape)