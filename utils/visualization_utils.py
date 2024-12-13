import numpy as np
import sys, os, yaml
import matplotlib.pyplot as plt
import wandb
import io
import torch
from PIL import Image
from tqdm import tqdm
import torch.distributed as dist

# Load configuration
config_path = '../config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

def extract_patches(image, patch_size):
    '''
    Extract patches from an image tensor.
    Args:
        image: Tensor of shape (C, H, W)
        patch_size: size of the patches
    Returns:
        patches: Tensor of shape (num_patches, C, patch_size, patch_size)
    '''
    # Unfold the image to get patches
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Rearrange dimensions to get patches in (num_patches, channels, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, image.shape[0], patch_size, patch_size)
    return patches

def assemble_patches_with_gaps(patches, gap_size, num_patches_per_row, patch_size, num_channels=3, depth=False):
    '''
    Assembles patches into an image with gaps between patches for cooler visualization.
    Args:
        patches: numpy array of shape (num_patches, num_channels, patch_size, patch_size)
        gap_size: size of the gap between patches (in pixels)
        num_patches_per_row: number of patches per row/column
        patch_size: size of each patch (assuming square patches)
        num_channels: number of channels in the image
        depth: whether it's a depth map (True) or RGB image (False)
    Returns:
        image_with_gaps: numpy array of shape (grid_size, grid_size, num_channels) or (grid_size, grid_size)
    '''
    #define the grid size as the size of the image with gaps
    grid_size = num_patches_per_row * patch_size + (num_patches_per_row - 1) * gap_size # in pixels
    if depth:
        #for depth use a single channel
        image_with_gaps = np.ones((grid_size, grid_size))
    else:
        #for RGB use 3 channels
        image_with_gaps = np.ones((grid_size, grid_size, num_channels))
    idx = 0
    #iterate over the patches and place them in the image with gaps
    for row in range(num_patches_per_row):
        for col in range(num_patches_per_row):
            #calculate the start position of the patch
            y_start = row * (patch_size + gap_size)
            x_start = col * (patch_size + gap_size)
            if depth:
                #for depth maps we only have one channel
                image_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size] = patches[idx]
            else:
                #for RGB images we have 3 channels
                image_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size, :] = patches[idx].transpose(1, 2, 0)
            idx += 1
    return image_with_gaps

def reshape_batch_of_masks(B, T, H, W, patch_size, num_patches, rgb_masks, depth_masks):
    rgb_masks_full = torch.zeros((B, T, H, W), dtype=rgb_masks.dtype, device=rgb_masks.device)
    depth_masks_full = torch.zeros((B, T, H, W), dtype=depth_masks.dtype, device=depth_masks.device)

    # Map the patches to the full frame
    patches_per_row = int(H / patch_size)  # 14 for a 224x224 image with 16x16 patches
    for b in range(B):
        for t in range(T):
            for patch_idx in range(num_patches**2):
                # Calculate the row and column in the 14x14 grid
                row = patch_idx // patches_per_row
                col = patch_idx % patches_per_row

                # Calculate the pixel indices in the full frame for this patch
                start_row = row * patch_size
                start_col = col * patch_size
                end_row = start_row + patch_size
                end_col = start_col + patch_size

                # Assign the mask value to the corresponding region of the full frame
                rgb_masks_full[b, t, start_row:end_row, start_col:end_col] = rgb_masks[b, t, patch_idx]
                depth_masks_full[b, t, start_row:end_row, start_col:end_col] = depth_masks[b, t, patch_idx]
    return rgb_masks_full, depth_masks_full


def log_visualizations(rgb_frames, depth_maps, reconstructed_image, reconstructed_depth, rgb_masks, depth_masks, epoch, batch_idx, prefix='Train'):
    '''
    Logs visualizations to WandB.
    Args:
        rgb_frames: input images (batch_size, 3, T, H, W)
        depth_maps: input depth maps (batch_size, 1, T, H, W)
        reconstructed_image: reconstructed RGB images [B, T, num_patches_per_frame, num_pixels_per_patch*channels]
        reconstructed_depth: reconstructed depth maps [B, T, num_patches_per_frame, num_pixels_per_patch*channels]
        rgb_masks: masks used during training for rgb frames [B, T, num_patches_per_frame]
        depth_masks: masks used during training for depth maps [B, T, num_patches_per_frame]
        epoch: current epoch
        batch_idx: current batch index
        prefix: 'Train' or 'Validation'
    '''
    rank = dist.get_rank()
    if rank != 0:
        return

    B, T, _ = rgb_masks.shape
    num_patches = 14 # TODO hardcoded
    patch_size = 16 # TODO hardcoded
    H, W = 224, 224 # TODO hardcoded

    reconstructed_image = reconstructed_image.view(B, T, num_patches, num_patches, patch_size**2 * 3)  # [B, T, num_patches_h, num_patches_w, num_pixels_per_patch * C]
    reconstructed_image = reconstructed_image.view(B, T, num_patches, num_patches, 3, patch_size, patch_size)  # last dimension to [C, patch_size_h, patch_size_w]
    reconstructed_image = reconstructed_image.permute(0, 4, 1, 2, 5, 3, 6).contiguous()  # [B, C, T, num_patches_h, patch_size_h, num_patches_w, patch_size_w]
    reconstructed_image = reconstructed_image.view(B, 3, T, H, W)
    
    reconstructed_depth = reconstructed_depth.view(B, T, num_patches, num_patches, patch_size**2)  # [B, T, num_patches_h, num_patches_w, num_pixels_per_patch]
    reconstructed_depth = reconstructed_depth.view(B, T, num_patches, num_patches, patch_size, patch_size)  # last dimension to [patch_size_h, patch_size_w]
    reconstructed_depth = reconstructed_depth.permute(0, 2, 4, 1, 5, 3).contiguous()  # [B, num_patches_h, patch_size_h, T, patch_size_w, num_patches_w]
    reconstructed_depth = reconstructed_depth.view(B, 1, T, H, W)

    # Verify the result
    rgb_mask, depth_mask = reshape_batch_of_masks(B, T, H, W, patch_size, num_patches, rgb_masks, depth_masks)

    # Select first frame of first batch
    rgb_mask = rgb_mask[0, 0].detach().cpu()
    depth_mask = depth_mask[0, 0].detach().cpu()

    depth_mean = config['data']['depth_stats']['mean']
    depth_std = config['data']['depth_stats']['std']
    img_size = config['model']['img_size']
    patch_size = config['model']['patch_size']
    
    # Select the first sample in the batch
    original_image = rgb_frames[0].detach().cpu()  # Shape: (3, T, H, W)
    original_depth = depth_maps[0].detach().cpu()  # Shape: (1, T, H, W)
    # rgb_mask = rgb_masks[0].detach().cpu()         # Shape: (3, T, H, W)
    # depth_mask = depth_masks[0].detach().cpu()     # Shape: (1, T, H, W)
    
    # Select the first frame in the sequence
    original_image = original_image[:, 0]  # Shape: (3, H, W)
    original_depth = original_depth[:, 0]  # Shape: (1, H, W)
    # rgb_mask = rgb_mask[:, 0]              # Shape: (3, H, W)
    # depth_mask = depth_mask[:, 0]          # Shape: (1, H, W)
    
    # Same for reconstructed image
    reconstructed_image = reconstructed_image[0].detach().cpu()[:, 0]  # Shape: (3, H, W)
    reconstructed_depth = reconstructed_depth[0].detach().cpu()[0, 0]  # Shape: (H, W)

    # Denormalize and normalize original RGB image
    original_rgb_image = original_image  # Shape: (3, H, W)
    
    # Denormalize depth map
    original_depth_map = original_depth.numpy() * depth_std + depth_mean
    original_depth_map_viz = (original_depth_map - original_depth_map.min()) / (original_depth_map.max() - original_depth_map.min() + 1e-8)
    
    # Extract patches
    original_patches_rgb = extract_patches(original_image, patch_size)  # Shape: (num_patches, 3, patch_size, patch_size)
    original_patches_depth = extract_patches(original_depth, patch_size)  # Shape: (num_patches, 1, patch_size, patch_size)

    # Denormalize patches
    original_patches_rgb_denorm = original_patches_rgb.numpy()
    original_patches_depth_denorm = original_patches_depth.numpy() * depth_std + depth_mean
    original_patches_depth_denorm = original_patches_depth_denorm.squeeze(1)  # Shape: (num_patches, patch_size, patch_size)
    
    # Normalize depth patches for visualization
    depth_min = original_patches_depth_denorm.min()
    depth_max = original_patches_depth_denorm.max()
    original_patches_depth_viz = (original_patches_depth_denorm - depth_min) / (depth_max - depth_min + 1e-8)
    
    # Assemble patches with gaps
    gap_size = 2
    num_patches_per_row = img_size // patch_size
    assembled_original_rgb = assemble_patches_with_gaps(original_patches_rgb_denorm, gap_size, num_patches_per_row, patch_size, num_channels=3)
    assembled_original_depth = assemble_patches_with_gaps(original_patches_depth_viz, gap_size, num_patches_per_row, patch_size, depth=True)

    # Visualize masks
    # rgb_mask_viz = rgb_mask.permute(1, 2, 0).numpy()  # Shape: (H, W, 3)
    # depth_mask_viz = depth_mask.squeeze(0).numpy()    # Shape: (H, W)
    
    # Normalize reconstructed depth map for visualization
    reconstructed_depth_map_viz = (reconstructed_depth - reconstructed_depth.min()) / (reconstructed_depth.max() - reconstructed_depth.min() + 1e-8)
    
    # Create depth images using matplotlib and save them to buffers
    depth_images = {}
    # Original Depth Map
    fig1 = plt.figure()
    plt.imshow(np.squeeze(original_depth_map_viz), cmap='viridis')
    plt.axis('off')
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig1)
    buf1.seek(0)
    depth_images[f'{prefix} Original Depth Map'] = wandb.Image(Image.open(buf1), caption='Original Depth Map')
    
    # Assembled Original Depth Patches
    fig2 = plt.figure()
    plt.imshow(np.squeeze(assembled_original_depth), cmap='viridis')
    plt.axis('off')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig2)
    buf2.seek(0)
    depth_images[f'{prefix} Assembled Original Depth Patches'] = wandb.Image(Image.open(buf2), caption='Original Depth Patches')
    
    # Reconstructed Depth Map
    fig3 = plt.figure()
    plt.imshow(reconstructed_depth_map_viz, cmap='viridis')
    plt.axis('off')
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig3)
    buf3.seek(0)
    depth_images[f'{prefix} Reconstructed Depth Map'] = wandb.Image(Image.open(buf3), caption='Reconstructed Depth Map')
    
    # Log images to WandB
    wandb.log({
        f'{prefix} Original RGB Image': wandb.Image(original_rgb_image),
        f'{prefix} Assembled Original RGB Patches': wandb.Image(assembled_original_rgb),
        f'{prefix} RGB Mask': wandb.Image(rgb_mask.to(torch.float32)),
        f'{prefix} Depth Mask': wandb.Image(depth_mask.to(torch.float32)),
        f'{prefix} Reconstructed RGB Image': wandb.Image(reconstructed_image),
        **depth_images,
        'Epoch': epoch,
        'Batch': batch_idx
    })
