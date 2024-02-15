from models import get_resnet, replace_bn_with_gn, ConditionalUnet1D
import torch
import torch.nn as nn 
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


pred_horizon = 16
obs_horizon = 4
action_horizon = 8

#@markdown ### **Network Demo**

# construct ResNet18 encoder
# if you have multiple camera views, use seperate encoder weights for each view.
vision_encoder_wrist = get_resnet('resnet34')
vision_encoder_overhead = get_resnet('resnet34')

# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
vision_encoder_wrist = replace_bn_with_gn(vision_encoder_wrist)
vision_encoder_overhead = replace_bn_with_gn(vision_encoder_overhead)

# ResNet18 has output dim of 512
vision_feature_dim = 512 * 2
# agent_pos is 7 dimensional
lowdim_obs_dim = 7
# observation feature has 519 dims in total per step
obs_dim = vision_feature_dim + lowdim_obs_dim
action_dim = 7

# create network object
noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder_wrist': vision_encoder_wrist,
    'vision_encoder_overhead': vision_encoder_overhead,
    'noise_pred_net': noise_pred_net
})


# try:
#     ckpt_path = "checkpoints/30ep_wrist_overhead.pt"
#     nets.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
#     print('Pretrained weights loaded before training')
# except:
#     print('Error while loading model before training')


# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = nets.to(device)





# training 
from tqdm import tqdm
import numpy as np 
from create_dataset import dataloader
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler


#@markdown ### **Training**
#@markdown
#@markdown Takes about 2.5 hours. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights

num_epochs = 20

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=nets.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nimage_wrist = nbatch['image_wrist'][:,:obs_horizon].to(device)
                nimage_overhead = nbatch['image_overhead'][:,:obs_horizon].to(device)
                nagent_pos = nbatch['joint_pos'][:,:obs_horizon].to(device)
                naction = nbatch['action'].to(device)
                B = nagent_pos.shape[0]

                # encoder vision features
                image_features_wrist = nets['vision_encoder_wrist'](
                    nimage_wrist.flatten(end_dim=1))
                image_features_wrist = image_features_wrist.reshape(
                    *nimage_wrist.shape[:2],-1)
                # (B,obs_horizon,D)

                # encoder vision features
                image_features_overhead = nets['vision_encoder_overhead'](
                    nimage_overhead.flatten(end_dim=1))
                image_features_overhead = image_features_overhead.reshape(
                    *nimage_overhead.shape[:2],-1)
                # (B,obs_horizon,D)

                # concatenate vision feature and low-dim obs
                obs_features = torch.cat([image_features_wrist, image_features_overhead, nagent_pos], dim=-1)
                obs_cond = obs_features.flatten(start_dim=1)
                # (B, obs_horizon * obs_dim)

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)
            
                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(nets.parameters())

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)
        tglobal.set_postfix(loss=np.mean(epoch_loss))

# Weights of the EMA model
# is used for inference
ema_nets = nets
ema.copy_to(ema_nets.parameters())


import os 
torch.save(ema_nets.state_dict(), '/home/meeroro/workspace/diffusionpolicy/checkpoints/reach_target/20ep_4obs.pt')

try:
    ckpt_path = "checkpoints/reach_target/20ep_4obs.pt"
    state_dict = torch.load(ckpt_path, map_location='cuda')
    ema_nets = nets
    ema_nets.load_state_dict(state_dict)
    print('Pretrained weights loaded.')
except:
    print('Error while loading model')
