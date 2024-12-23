import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
 

def plot_patches(ax, patches, patch_size, title, cmap=None):
    """
    Plots a row of patches on the given axes.

    Args:
        ax: The matplotlib axes to plot on.
        patches: The array of patches [num_patches, patch_size^2 * channels].
        patch_size: The height/width of each patch.
        title: Title for the row of patches.
        cmap: Colormap to use for grayscale images.
    """
    num_patches = patches.shape[0]
    num_channels = patches.shape[1] // (patch_size ** 2)
    for i in range(num_patches):
        patch = patches[i].astype(np.float32).reshape(num_channels, patch_size, patch_size)  # Convert to float32
        patch = patch.transpose(1, 2, 0)
        if num_channels == 1:  # Depth (grayscale)
            patch = patch.squeeze(-1)  # Remove the channel dimension
            ax[i].imshow(patch, cmap=cmap)
        else:  # RGB
            ax[i].imshow(patch)
        ax[i].axis('off')
    ax[0].set_title(title)


def save_plots(rgb_orig_patches, rgb_recon_patches, depth_orig_patches, depth_recon_patches, save_path="patch_visualization.png"):
    """
    Saves the original and reconstructed patches for the first frame of the first batch to a file.

    Args:
        rgb_orig_patches: Original RGB patches [num_masked, C*patch_size^2]
        rgb_recon_patches: Reconstructed RGB patches [num_masked, C*patch_size^2]
        depth_orig_patches: Original Depth patches [num_masked, C*patch_size^2]
        depth_recon_patches: Reconstructed Depth patches [num_masked, C*patch_size^2]
        save_path: Path to save the plot image.
    """

    print('Shape rgb_orig_patches:', rgb_orig_patches.shape)
    print('Shape rgb_recon_patches:', rgb_recon_patches.shape)
    print('Shape depth_orig_patches:', depth_orig_patches.shape)
    print('Shape depth_recon_patches:', depth_recon_patches.shape)
    
    print('RGB ORIGINAL PATCHES:\n', rgb_orig_patches)
    print('RGB RECON PATCHES:\n', rgb_recon_patches)

    # Parameters
    patch_size = int(np.sqrt(rgb_orig_patches.shape[1] // 3))  # Calculate patch size from RGB patch shape
    num_patches_to_plot = 60

    # Subset patches for plotting
    # take the patches in the middle
    rgb_orig_patches = rgb_orig_patches[rgb_orig_patches.shape[0]//2 - num_patches_to_plot//2:rgb_orig_patches.shape[0]//2 + num_patches_to_plot//2]
    rgb_recon_patches = rgb_recon_patches[rgb_recon_patches.shape[0]//2 - num_patches_to_plot//2:rgb_recon_patches.shape[0]//2 + num_patches_to_plot//2]
    depth_orig_patches = depth_orig_patches[depth_orig_patches.shape[0]//2 - num_patches_to_plot//2:depth_orig_patches.shape[0]//2 + num_patches_to_plot//2]
    depth_recon_patches = depth_recon_patches[depth_recon_patches.shape[0]//2 - num_patches_to_plot//2:depth_recon_patches.shape[0]//2 + num_patches_to_plot//2]

    # Create figure
    fig, axs = plt.subplots(4, num_patches_to_plot, figsize=(num_patches_to_plot * 2, 8))

    # RGB Original Patches
    plot_patches(axs[0], rgb_orig_patches, patch_size, "RGB Original", cmap=None)

    # RGB Reconstructed Patches
    plot_patches(axs[1], rgb_recon_patches, patch_size, "RGB Reconstructed", cmap=None)

    # Depth Original Patches
    depth_patch_size = int(np.sqrt(depth_orig_patches.shape[1]))  # Calculate depth patch size
    plot_patches(axs[2], depth_orig_patches, depth_patch_size, "Depth Original", cmap='viridis')

    # Depth Reconstructed Patches
    plot_patches(axs[3], depth_recon_patches, depth_patch_size, "Depth Reconstructed", cmap='viridis')

    # Adjust layout
    fig.tight_layout()

    # Save to file
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Patch visualization saved to {save_path}")


def compute_loss(rgb_original, depth_original,
                rgb_frame_reconstruction, depth_frame_reconstruction,
                rgb_masks_per_frame, depth_masks_per_frame,
                patch_size=16,
                alpha=1.0,
                beta=1.0):
    '''
    Compute the reconstruction loss for the RGB and depth frames.

    Args:
        rgb_original: original RGB frames [B, 3, T, H, W]
        depth_original: original depth maps [B, 1, T, H, W]
        rgb_frame_reconstruction: reconstructed RGB frames [B, T, num_patches_per_frame, num_pixels_per_patch*3]
        depth_frame_reconstruction: reconstructed depth maps [B, T, num_patches_per_frame, num_pixels_per_patch*1]
        rgb_masks_per_frame: masks used during training for rgb frames [B, T, num_patches_per_frame]
        depth_masks_per_frame: masks used during training for depth maps [B, T, num_patches_per_frame]
        patch_size: size of the patches (default=16)
        alpha: weight for RGB loss
        beta: weight for Depth loss

    Returns:
        rgb_loss: RGB reconstruction loss
        depth_loss: Depth reconstruction loss
        total_loss: Total loss as a weighted sum of RGB and Depth losses
    '''
    # Print all input shapes
    print('LOSS INPUT SHAPES ----------------------')
    print('Shape rgb_original:', rgb_original.shape)
    print('Shape depth_original:', depth_original.shape)
    print('Shape rgb_frame_reconstruction:', rgb_frame_reconstruction.shape)
    print('Shape depth_frame_reconstruction:', depth_frame_reconstruction.shape)
    print('Shape rgb_masks_per_frame:', rgb_masks_per_frame.shape)
    print('Shape depth_masks_per_frame:', depth_masks_per_frame.shape)
    print('----------------------------------------')

    B, C_rgb, T, H, W = rgb_original.shape
    B_d, C_d, T_d, H_d, W_d = depth_original.shape
    assert B == B_d and T == T_d and H == H_d and W == W_d, "Input tensors must have the same shape"

    # Determine patch-related parameters
    num_patches_per_frame = rgb_frame_reconstruction.shape[2]  # e.g., 196
    num_pixels_per_patch_rgb = rgb_frame_reconstruction.shape[-1] // C_rgb  # e.g., 768 // 3 = 256
    patch_size_rgb = int(np.sqrt(num_pixels_per_patch_rgb))  # e.g., 16
    num_pixels_per_patch_depth = depth_frame_reconstruction.shape[-1] // C_d  # e.g., 256 // 1 = 256
    patch_size_depth = int(np.sqrt(num_pixels_per_patch_depth))  # e.g., 16

    assert patch_size_rgb == patch_size_depth, "RGB and Depth patch sizes must match"
    patch_size = patch_size_rgb

    # Reshape RGB original frames to [B, T, num_patches, C * patch_size^2]
    rgb_perm = rgb_original.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
    rgb_reshaped = rgb_perm.reshape(B * T, C_rgb, H, W)  # [B*T, C, H, W]
    rgb_patches = F.unfold(rgb_reshaped, kernel_size=patch_size, stride=patch_size)  # [B*T, C*patch_size^2, num_patches]
    rgb_patches = rgb_patches.transpose(1, 2).reshape(B, T, num_patches_per_frame, C_rgb * patch_size * patch_size)  # [B, T, num_patches, C*patch_size^2]

    print('Shape rgb_patches:', rgb_patches.shape)

    # Reshape Depth original frames to [B, T, num_patches, C * patch_size^2]
    depth_perm = depth_original.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
    depth_reshaped = depth_perm.reshape(B * T, C_d, H_d, W_d)  # [B*T, C, H, W]
    depth_patches = F.unfold(depth_reshaped, kernel_size=patch_size, stride=patch_size)  # [B*T, C*patch_size^2, num_patches]
    depth_patches = depth_patches.transpose(1, 2).reshape(B, T, num_patches_per_frame, C_d * patch_size * patch_size)  # [B, T, num_patches, C*patch_size^2]

    print('Shape depth_patches:', depth_patches.shape)

    # Masks: [B, T, num_patches]
    rgb_mask = rgb_masks_per_frame.bool()  # Convert to boolean mask
    depth_mask = depth_masks_per_frame.bool()

    print('Shape rgb_mask:', rgb_mask.shape)
    print('Shape depth_mask:', depth_mask.shape)

    # Select only the masked patches
    # RGB
    rgb_recon_masked = rgb_frame_reconstruction[rgb_mask]  # [num_masked, C*patch_size^2]
    rgb_orig_masked = rgb_patches[rgb_mask]  # [num_masked, C*patch_size^2]

    print('Shape rgb_recon_masked:', rgb_recon_masked.shape)
    print('Shape rgb_orig_masked:', rgb_orig_masked.shape)

    # Depth
    depth_recon_masked = depth_frame_reconstruction[depth_mask]  # [num_masked, C*patch_size^2]
    depth_orig_masked = depth_patches[depth_mask]  # [num_masked, C*patch_size^2]

    # Compute MSE loss for RGB
    if rgb_recon_masked.numel() > 0:
        rgb_loss = F.mse_loss(rgb_recon_masked, rgb_orig_masked)
    else:
        rgb_loss = torch.tensor(0.0, device=rgb_original.device)

    # Compute MSE loss for Depth
    if depth_recon_masked.numel() > 0:
        depth_loss = F.mse_loss(depth_recon_masked, depth_orig_masked)
    else:
        depth_loss = torch.tensor(0.0, device=depth_original.device)

    # Compute total loss as a weighted sum
    total_loss = alpha * rgb_loss + beta * depth_loss

    save_plots(rgb_orig_masked.cpu().detach().numpy(),
               rgb_recon_masked.cpu().detach().numpy(),
               depth_orig_masked.cpu().detach().numpy(),
               depth_recon_masked.cpu().detach().numpy())

    return rgb_loss, depth_loss, total_loss

if __name__ == "__main__":
    """
    Testing the compute_loss function with mock inputs.
    """
    # Configuration parameters
    B = 2  # Batch size
    C_rgb = 3
    C_depth = 1
    T = 16
    H = 224
    W = 224
    patch_size = 16
    alpha = 1.0
    beta = 1.0

    # Determine number of patches per frame
    num_patches_per_frame = (H // patch_size) * (W // patch_size)  # e.g., 14 * 14 = 196

    # Generate ground truth tensors as zeros for clearer loss interpretation
    torch.manual_seed(0)  # For reproducibility
    rgb_original = torch.zeros(B, C_rgb, T, H, W)
    depth_original = torch.zeros(B, C_depth, T, H, W)

    # Generate masks (randomly mask 25% of the patches)
    mask_ratio = 0.25
    num_masked_patches = int(num_patches_per_frame * mask_ratio)
    rgb_masks_per_frame = torch.zeros(B, T, num_patches_per_frame, dtype=torch.long)
    depth_masks_per_frame = torch.zeros(B, T, num_patches_per_frame, dtype=torch.long)

    for b in range(B):
        for t in range(T):
            masked_indices = torch.randperm(num_patches_per_frame)[:num_masked_patches]
            rgb_masks_per_frame[b, t, masked_indices] = 1
            depth_masks_per_frame[b, t, masked_indices] = 1

    # Create two sets of reconstructed tensors
    # Reconstruction 1: Small Gaussian noise
    noise_small = 0.01
    rgb_reconstruction_small = rgb_original.clone()
    depth_reconstruction_small = depth_original.clone()

    # Apply small noise only to masked patches
    rgb_reconstruction_small = rgb_reconstruction_small.view(B, C_rgb, T, -1)  # [B, C, T, H*W]
    depth_reconstruction_small = depth_reconstruction_small.view(B, C_depth, T, -1)  # [B, C, T, H*W]

    for b in range(B):
        for t in range(T):
            for p in range(num_patches_per_frame):
                if rgb_masks_per_frame[b, t, p]:
                    start = p * patch_size * patch_size
                    end = start + patch_size * patch_size
                    rgb_reconstruction_small[b, :, t, start:end] += noise_small * torch.randn_like(rgb_reconstruction_small[b, :, t, start:end])
                if depth_masks_per_frame[b, t, p]:
                    start = p * patch_size * patch_size
                    end = start + patch_size * patch_size
                    depth_reconstruction_small[b, :, t, start:end] += noise_small * torch.randn_like(depth_reconstruction_small[b, :, t, start:end])

    # Reconstruction 2: Large Gaussian noise
    noise_large = 0.1
    rgb_reconstruction_large = rgb_original.clone()
    depth_reconstruction_large = depth_original.clone()

    # Apply large noise only to masked patches
    rgb_reconstruction_large = rgb_reconstruction_large.view(B, C_rgb, T, -1)  # [B, C, T, H*W]
    depth_reconstruction_large = depth_reconstruction_large.view(B, C_depth, T, -1)  # [B, C, T, H*W]

    for b in range(B):
        for t in range(T):
            for p in range(num_patches_per_frame):
                if rgb_masks_per_frame[b, t, p]:
                    start = p * patch_size * patch_size
                    end = start + patch_size * patch_size
                    rgb_reconstruction_large[b, :, t, start:end] += noise_large * torch.randn_like(rgb_reconstruction_large[b, :, t, start:end])
                if depth_masks_per_frame[b, t, p]:
                    start = p * patch_size * patch_size
                    end = start + patch_size * patch_size
                    depth_reconstruction_large[b, :, t, start:end] += noise_large * torch.randn_like(depth_reconstruction_large[b, :, t, start:end])

    # Reshape reconstructions back to [B, T, num_patches_per_frame, C * patch_size^2]
    rgb_reconstruction_small = rgb_reconstruction_small.view(B, C_rgb, T, H * W).permute(0, 2, 1, 3).reshape(B, T, C_rgb * patch_size * patch_size, num_patches_per_frame).permute(0,1,3,2)  # [B, T, num_patches, C*patch_size^2]
    depth_reconstruction_small = depth_reconstruction_small.view(B, C_depth, T, H * W).permute(0, 2, 1, 3).reshape(B, T, C_depth * patch_size * patch_size, num_patches_per_frame).permute(0,1,3,2)  # [B, T, num_patches, C*patch_size^2]

    rgb_reconstruction_large = rgb_reconstruction_large.view(B, C_rgb, T, H * W).permute(0, 2, 1, 3).reshape(B, T, C_rgb * patch_size * patch_size, num_patches_per_frame).permute(0,1,3,2)  # [B, T, num_patches, C*patch_size^2]
    depth_reconstruction_large = depth_reconstruction_large.view(B, C_depth, T, H * W).permute(0, 2, 1, 3).reshape(B, T, C_depth * patch_size * patch_size, num_patches_per_frame).permute(0,1,3,2)  # [B, T, num_patches, C*patch_size^2]

    # Compute loss for small noise reconstruction
    rgb_loss_small, depth_loss_small, total_loss_small = compute_loss(
        rgb_original=rgb_original,
        depth_original=depth_original,
        rgb_frame_reconstruction=rgb_reconstruction_small,
        depth_frame_reconstruction=depth_reconstruction_small,
        rgb_masks_per_frame=rgb_masks_per_frame,
        depth_masks_per_frame=depth_masks_per_frame,
        patch_size=patch_size,
        alpha=alpha,
        beta=beta
    )

    # Compute loss for large noise reconstruction
    rgb_loss_large, depth_loss_large, total_loss_large = compute_loss(
        rgb_original=rgb_original,
        depth_original=depth_original,
        rgb_frame_reconstruction=rgb_reconstruction_large,
        depth_frame_reconstruction=depth_reconstruction_large,
        rgb_masks_per_frame=rgb_masks_per_frame,
        depth_masks_per_frame=depth_masks_per_frame,
        patch_size=patch_size,
        alpha=alpha,
        beta=beta
    )

    # Print the results
    print("Loss Comparison:")
    print(f"RGB Loss (Small Noise): {rgb_loss_small.item():.6f}")
    print(f"RGB Loss (Large Noise): {rgb_loss_large.item():.6f}")
    print(f"Depth Loss (Small Noise): {depth_loss_small.item():.6f}")
    print(f"Depth Loss (Large Noise): {depth_loss_large.item():.6f}")
    print(f"Total Loss (Small Noise): {total_loss_small.item():.6f}")
    print(f"Total Loss (Large Noise): {total_loss_large.item():.6f}")

    # Assertions to ensure that larger noise results in higher loss
    assert rgb_loss_large > rgb_loss_small, "RGB loss with large noise should be greater than with small noise."
    assert depth_loss_large > depth_loss_small, "Depth loss with large noise should be greater than with small noise."
    assert total_loss_large > total_loss_small, "Total loss with large noise should be greater than with small noise."

    print("\nAll assertions passed. The loss function behaves as expected.")