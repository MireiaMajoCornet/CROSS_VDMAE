'''Import Libraries'''
import os
import torch
import sys
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import yaml
import torch.optim as optim
import wandb
from tqdm import tqdm
import torch.multiprocessing as mp

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from tubeletembed import TubeletEmbed
from loss import compute_loss
from visualization_utils import log_visualizations

sys.path.append(os.path.join(os.path.dirname(__file__), '../models'))
from videomae_cross_modal import CrossModalVideoMAE

sys.path.append(os.path.join(os.path.dirname(__file__), '../data'))
from dataloader import VideoFrameDataset


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

def main():

    '''Set Up Configuration'''
    # Get rank and world_size from environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])  # Use this for device selection
    
    # Load configuration
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    wandb_config = config['wandb']

    # Set the device for the current process
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)  # Use the device variable
    
    #Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')

    # Initialize the model
    model = CrossModalVideoMAE(model_config).to(device)
    
    # Wrap the model for distributed training
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)  # Wrap model for DDP
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    else:
        model = model

    # Compile the model with torch for faster performance
    #model = torch.compile(model)

    '''Set up DepthAnythingV2 Model'''
    sys.path.append(os.path.join(os.path.dirname(__file__), '../depth_anything_v2'))
    from depth_anything_v2.dpt import DepthAnythingV2
    
    #Initialize the depth model
    model_configs_depth = {
        'vitl': {
            'encoder': 'vitl',
            'features': 256,
            'out_channels': [256, 512, 1024, 1024]
        }
    }

    depth_model = DepthAnythingV2(**model_configs_depth['vitl'])
    depth_model = depth_model.to(device)
    depth_model.eval()
    
    # Set it into evaluation mode
    with torch.no_grad():
        depth_model.load_state_dict(torch.load(data_config['depth_model_checkpoint'], map_location=device))
    
    '''Set Up Dataset and Dataloader'''
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((model_config['img_size'], model_config['img_size'])),
        transforms.ToTensor()
    ])

    # Initialize the dataset
    video_folder = data_config['finevideo_path'] if os.path.exists(data_config['finevideo_path']) else '/home/ndelafuente/VD-MAE/sports_videos'

    #single video path if we want to overfit to check if the model is learning
    single_video_path = data_config['single_video_path'] 

    dataset = VideoFrameDataset(
            video_folder = video_folder,
            transform = transform,
            depth_model=depth_model,
            num_frames = model_config['num_frames'],
            frame_interval = model_config['frame_interval'],
            single_video_path = single_video_path
        )
    
    # Initialize the distributed sampler for the dataset 
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)  # DistributedSampler for data loading

    # Initialize the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        sampler=sampler,
        shuffle=False,
        num_workers=training_config['num_workers'],
        pin_memory=True 
    )

    '''Set Up Optimizer and Scheduler'''
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay']
    )

    # Define lr scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config['num_epochs'])

    '''Set Up Weights and Biases'''
    if rank == 0:
        print(f"Rank: {rank}, Initializing wandb")
        wandb.init(entity=wandb_config['entity'], project=wandb_config['project'], name=wandb_config['name'])
        wandb.watch(model, log='all', log_freq=wandb_config['log_interval'])

    '''Training Loop'''
    num_epochs = training_config['num_epochs']
    mask_ratio = training_config['mask_ratio']

    scaler = torch.amp.GradScaler('cuda')  # Mixed precision training for optimization
    best_loss = float('inf')  # Set the best loss to infinity

    for epoch in range(num_epochs):
        
        # Set the model to train mode
        model.train()

        # Set the total losses to 0
        total_loss = 0.0
        total_rgb_loss = 0.0
        total_depth_loss = 0.0

        # Set epoch for DistributedSampler
        dataloader.sampler.set_epoch(epoch)

        # Set the progress bar
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        print(f"Length of dataloader: {len(dataloader)}")  # If distributed training, this will be the number of batches per process
        
        # Iterate over the dataloader
        for batch_idx, batch in progress_bar:
            rgb_frames, _, depth_maps, _, _ = batch

            # Send data to device
            rgb_frames = rgb_frames.to(device)
            depth_maps = depth_maps.to(device)

            # Batch size
            batch_size = config['training']['batch_size']
            assert batch_size == rgb_frames.size(0) == depth_maps.size(0), "Batch size mismatch"
            
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.amp.autocast('cuda'):
                # Get the reconstructions for the frames and depth maps
                # TODO maybe the model could also return
                # only the reconstruction of the masked patches
                # so we can compute the loss only on them
                rgb_frame_reconstruction, depth_frame_reconstruction, rgb_masks_per_frame, depth_masks_per_frame  = model(rgb_frames, depth_maps)

                # Calculate the losses
                rgb_loss, depth_loss, loss = compute_loss(rgb_frames, 
                                                        depth_maps,
                                                        rgb_frame_reconstruction,
                                                        depth_frame_reconstruction, 
                                                        rgb_masks_per_frame,
                                                        depth_masks_per_frame,
                                                        patch_size=model_config['patch_size'],
                                                        alpha=training_config['alpha'],
                                                        beta=training_config['beta'])
                
            print(f"Loss: {loss}, Type: {type(loss)}")
            print(f"Loss device: {loss.device}")
            print(f"Model outputs: rgb_frame_reconstruction device: {rgb_frame_reconstruction.device}, depth_frame_reconstruction device: {depth_frame_reconstruction.device}")

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Update the total losses
            total_loss += loss.item()
            total_rgb_loss += rgb_loss.item()
            total_depth_loss += depth_loss.item()
            
            # Log metrics to wandb every N batches
            if rank == 0 and batch_idx % wandb_config['log_interval'] == 0:
                log_visualizations(rgb_frames,
                                    depth_maps,
                                    rgb_frame_reconstruction,
                                    depth_frame_reconstruction,
                                    rgb_masks_per_frame,
                                    depth_masks_per_frame,
                                    epoch,
                                    prefix='Train')
            
        # Step the scheduler
        scheduler.step()
        
        # Average the total losses
        avg_loss = total_loss / len(dataloader)
        avg_rgb_loss = total_rgb_loss / len(dataloader)
        avg_depth_loss = total_depth_loss / len(dataloader)

        if rank == 0:
            # Log epoch metrics to wandb and console
            wandb.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'rgb_loss': avg_rgb_loss,
                'depth_loss': avg_depth_loss
            })
            
            # Save the model checkpoint if the loss is the best so far
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': best_loss,
                }, os.path.join(training_config['model_checkpoint_path'], 'model_at_epoch_{}.pth'))
            
            print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f} RGB Loss: {avg_rgb_loss:.4f} Depth Loss: {avg_depth_loss:.4f}")
    
    dist.destroy_process_group() # Destroy the process group after training

if __name__ == "__main__":
    main()