# models/videomae_cross_modal.py

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.layers import *

from functools import partial
import os
import sys
import logging
import math

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from tubeletembed import TubeletEmbed
from round_masking import round_masking

# Import visualization libraries and set backend for clusters without display
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Configure the logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level to INFO




class CrossModalVideoMAE(nn.Module):
    ''' Cross-modal VideoMAE model for Masked Video Reconstruction '''
    def __init__(self, config):
        super(CrossModalVideoMAE, self).__init__()
        
        # Initialize RGB tubelet embedding layer
        self.rgb_tubelet_embed = TubeletEmbed(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            num_frames=config['num_frames'],
            tubelet_size=config['tubelet_size'],
            in_chans=3,
            embed_dim=config['embed_dim']
        )
        
        # Initialize the Depth tubelet embedding layer
        self.depth_tubelet_embed = TubeletEmbed(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            num_frames=config['num_frames'],
            tubelet_size=config['tubelet_size'],
            in_chans=1,
            embed_dim=config['embed_dim']
        )
        
        # Get the number of tubelets from the tubelet embedding layer
        num_tubelets = self.rgb_tubelet_embed.num_tubelets  # Same number of patches for RGB and Depth
        
        # Initialize positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tubelets, config['embed_dim']))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize the transformer encoder layers
        self.encoder = nn.ModuleList([
            Block(
                dim=config['embed_dim'],
                num_heads=config['encoder_num_heads'],
                mlp_ratio=config['encoder_mlp_ratio'],
                qkv_bias=True,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU
            ) for _ in range(config['num_layers_encoder'])
        ])

        # Normalize the encoder output
        self.encoder_norm = nn.LayerNorm(config['embed_dim'], eps=1e-6)
        
        # Initialize two separate decoders for RGB and Depth
        
        # RGB decoder
        self.rgb_decoder = nn.ModuleList([
            Block(
                dim=config['decoder_embed_dim'],
                num_heads=config['decoder_num_heads'],
                mlp_ratio=config['decoder_mlp_ratio'],
                qkv_bias=True,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU
            ) for _ in range(config['num_layers_decoder'])
        ])
        # Normalize the RGB decoder output
        self.rgb_decoder_norm = nn.LayerNorm(config['decoder_embed_dim'], eps=1e-6)
        # Output layer for RGB frames
        self.rgb_head = nn.Linear(config['decoder_embed_dim'], config['tubelet_size'] * config['patch_size'] ** 2 * 3)
        
        # Depth decoder
        self.depth_decoder = nn.ModuleList([
            Block(
                dim=config['decoder_embed_dim'],
                num_heads=config['decoder_num_heads'],
                mlp_ratio=config['decoder_mlp_ratio'],
                qkv_bias=True,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU
            )
            for _ in range(config['num_layers_decoder'])
        ])
        # Normalize the Depth decoder output
        self.depth_decoder_norm = nn.LayerNorm(config['decoder_embed_dim'], eps=1e-6)
        # Output layer for Depth frames
        self.depth_head = nn.Linear(config['decoder_embed_dim'], config['tubelet_size'] * config['patch_size'] ** 2 * 1)
        
        # Initialize the mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config['embed_dim'])) 
        #nn.init.trunc_normal_(self.mask_token, std=0.02)
      
        # Loss weighting factors
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 1.0)

        # Get embed dimensions
        self.embed_dim = config['embed_dim']
        self.decoder_embed_dim = config['decoder_embed_dim']
        
        # Function to project encoder output to decoder input over the last dimension
        self.encoder_to_decoder = nn.Linear(self.embed_dim, self.decoder_embed_dim)  # [B, N, embed_dim] -> [B, N, decoder_embed_dim]
        
        # Store necessary config parameters as class attributes
        self.img_size = config['img_size']
        self.patch_size = config['patch_size']
        self.num_frames = config['num_frames']
        self.tubelet_size = config['tubelet_size']
        self.mask_ratio = round_masking(config['mask_ratio'])

        #num_patches_per_frame = (self.img_size // self.patch_size) ** 2
        #assert (num_patches_per_frame * self.mask_ratio).is_integer(), "Number of patches per frame must be divisible by the mask ratio, CHANGE YOUR MASK RATIO"
        
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, rgb_frames, depth_maps):
        '''
        Forward pass of the CrossModalVideoMAE model.

        Parameters:
            - rgb_frames: Tensor of shape [B, 3, T, H, W]
            - depth_maps: Tensor of shape [B, 1, T, H, W]

        Returns:
            - rgb_reconstruction: Reconstructed RGB patches.
            - depth_reconstruction: Reconstructed depth patches.
        '''
        B, C, T, H, W = rgb_frames.shape
        assert C == 3, "Input RGB tensor must have 3 channels"

        _, C_d, T_d, H_d, W_d = depth_maps.shape
        assert C_d == 1, "Input Depth tensor must have 1 channel"
        assert T == T_d and H == H_d and W == W_d, "RGB and Depth tensors must have the same dimensions"

        # Apply tubelet embedding to the RGB and Depth frames
        rgb_embed = self.rgb_tubelet_embed(rgb_frames)  # Shape: [B, N, embed_dim]
        depth_embed = self.depth_tubelet_embed(depth_maps)  # Shape: [B, N, embed_dim]

        # Generate masks inside the forward method
        N = rgb_embed.size(1)  # Total number of tubelets across all frames
        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        num_patches_per_frame = H_patches * W_patches
        num_temporal_positions = T // self.tubelet_size
        num_tubelets = num_temporal_positions * num_patches_per_frame
        assert N == num_tubelets, "Number of tubelets mismatch"
        num_masks_per_temporal_position = int(num_patches_per_frame * self.mask_ratio)

        # Generate masks for RGB
        rgb_masks = torch.zeros(B, num_tubelets, dtype=torch.bool, device=rgb_frames.device)
        for b in range(B):
            for t_pos in range(num_temporal_positions):
                start_idx = t_pos * num_patches_per_frame
                indices = torch.randperm(num_patches_per_frame, device=rgb_frames.device)[:num_masks_per_temporal_position]
                rgb_masks[b, start_idx + indices] = True  # Apply the masks

        # Generate masks for Depth
        depth_masks = torch.zeros(B, num_tubelets, dtype=torch.bool, device=depth_maps.device)
        for b in range(B):
            for t_pos in range(num_temporal_positions):
                start_idx = t_pos * num_patches_per_frame
                indices = torch.randperm(num_patches_per_frame, device=depth_maps.device)[:num_masks_per_temporal_position]
                depth_masks[b, start_idx + indices] = True  # Apply the masks

        # For visualization, store the masks separately
        self.rgb_masks = rgb_masks
        self.depth_masks = depth_masks
        logger.info(f'RGB Masks shape: {rgb_masks.shape}')
        #logger.info(f'RGB Masks: {rgb_masks}')
        logger.info(f'Depth Masks shape: {depth_masks.shape}')
        #logger.info(f'Depth Masks: {depth_masks}')

        # Apply masking to RGB and Depth embeddings
        mask_tokens_rgb = self.mask_token.expand(B, N, -1)  # Shape: [B, N, embed_dim]
        mask_tokens_depth = self.mask_token.expand(B, N, -1)  # Shape: [B, N, embed_dim]
        logger.info(f'Mask Tokens RGB shape: {mask_tokens_rgb.shape}')
        logger.info(f'Mask Tokens Depth shape: {mask_tokens_depth.shape}')

        # Add positional encoding
        rgb_embed += self.pos_embed
        depth_embed += self.pos_embed
        logger.info(f'RGB Embeddings + Pos Enc shape: {rgb_embed.shape}')
        #logger.info(f'RGB Embeddings + Pos Enc: {rgb_embed}')
        logger.info(f'Depth Embeddings + Pos Enc shape: {depth_embed.shape}')
        #logger.info(f'Depth Embeddings + Pos Enc: {depth_embed}')

        #send the masks to the device
        rgb_masks = rgb_masks.to(self.device)
        depth_masks = depth_masks.to(self.device)
        rgb_embed = rgb_embed.to(self.device)
        depth_embed = depth_embed.to(self.device)
        mask_tokens_rgb = mask_tokens_rgb.to(self.device)
        mask_tokens_depth = mask_tokens_depth.to(self.device)

        #set as rgb embedding only the visible patches by doing torch.where(rgb_masks.unsqueeze(-1), rgb_embed, mask_tokens_rgb)
        rgb_embeddings = torch.where(rgb_masks.unsqueeze(-1), mask_tokens_rgb, rgb_embed)
        depth_embeddings = torch.where(depth_masks.unsqueeze(-1), mask_tokens_depth, depth_embed)
        logger.info(f'RGB Embeddings shape: {rgb_embeddings.shape}')
        #logger.info(f'RGB Embeddings: {rgb_embeddings}')
        logger.info(f'Depth Embeddings shape: {depth_embeddings.shape}')
        #logger.info(f'Depth Embeddings: {depth_embeddings}')

        # Remove the zero entries (use mask to filter out the tubelets)
        rgb_masks = rgb_masks.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        depth_masks = depth_masks.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        logger.info(f'RGB Masks shape (zeros removed): {rgb_masks.shape}')
        logger.info(f'Depth Masks shape (zeros removed): {depth_masks.shape}')
        

        print(f'RGB Embeddings shape of visible tubelets: {rgb_embeddings[~rgb_masks.bool()].shape}')
        print(f'Depth Embeddings shape of visible tubelets: {depth_embeddings[~depth_masks.bool()].shape}')

        
        print(f'(num_tubelets*(1-self.mask_ratio)): {num_tubelets*(1-self.mask_ratio)}')
        
        print(f'RGB Embeddings shape of visible tubelets view [B, N_tubelets, embed_dim]: {rgb_embeddings[~rgb_masks.bool()].view(B, int(num_tubelets*(1-self.mask_ratio)), self.embed_dim).shape}')
        print(f'Depth Embeddings shape of visible tubelets view [B, N_tubelets, embed_dim]: {depth_embeddings[~depth_masks.bool()].view(B, int(num_tubelets*(1-self.mask_ratio)), self.embed_dim).shape}')

        only_visible_embedded_tubelets_rgb = rgb_embeddings[~rgb_masks.bool()].view(B, int(num_tubelets*(1-self.mask_ratio)), self.embed_dim)
        only_visible_embedded_tubelets_depth = depth_embeddings[~depth_masks.bool()].view(B, int(num_tubelets*(1-self.mask_ratio)), self.embed_dim)
        logger.info(f'Only Visible Embedded Tubelets RGB shape: {only_visible_embedded_tubelets_rgb.shape}')
        #logger.info(f'Only Visible Embedded Tubelets RGB: {only_visible_embedded_tubelets_rgb}')
        logger.info(f'Only Visible Embedded Tubelets Depth shape: {only_visible_embedded_tubelets_depth.shape}')
        #logger.info(f'Only Visible Embedded Tubelets Depth: {only_visible_embedded_tubelets_depth}')

        rgb_embeddings = only_visible_embedded_tubelets_rgb
        depth_embeddings = only_visible_embedded_tubelets_depth

        # Encode RGB and Depth separately
        for block in self.encoder:
            rgb_embeddings = block(rgb_embeddings)
            depth_embeddings = block(depth_embeddings)

        # Normalize encoder outputs
        encoded_rgb_embeddings = self.encoder_norm(rgb_embeddings)
        encoded_depth_embeddings = self.encoder_norm(depth_embeddings)
        
        '''NOW ADD THE MASKED TOKENS TO THEIR RESPECTIVE POSITION BEFORE DECODING'''
        # Assert that the number of zeros in masks_tokens_rgb is equal to N_tubelets * B * Embed_dim
        zeros_in_mask = (rgb_masks == 0).sum().item()
        N_total = encoded_rgb_embeddings.size(1) * encoded_rgb_embeddings.size(0) * encoded_rgb_embeddings.size(2) 
        logger.info(f'Number of zeros in mask: {zeros_in_mask}')
        logger.info(f'N: {N_total}')
        assert zeros_in_mask == N_total, f"Expected {N_total} zeros, but found {zeros_in_mask} zeros in the mask."

        # when you find the first zero in rgb_masks, insert the first element of encoded_rgb_embeddings, then if the mask contains a 1, keep a 0 in the mask, and so on:
        rgb_encoded_with_mask = torch.zeros(B, num_tubelets, self.embed_dim, device=encoded_rgb_embeddings.device)
        logger.info(f'RGB Encoded with Mask shape when created: {rgb_encoded_with_mask.shape}')

        depth_encoded_with_mask = torch.zeros(B, num_tubelets, self.embed_dim, device=encoded_depth_embeddings.device)
        logger.info(f'Depth Encoded with Mask shape when created: {depth_encoded_with_mask.shape}')

        logger.info(f'RGB Masks shape: {rgb_masks.shape}')
        #logger.info(f'RGB Masks: {rgb_masks}')
        logger.info(f'RGB Masks batch 0 shape: {rgb_masks[0].shape}')
        #logger.info(f'RGB Masks batch 0: {rgb_masks[0]}')
        logger.info(f'RGB Masks batch 0 tubelet 0 shape: {rgb_masks[0, 0].shape}')
        #logger.info(f'RGB Masks batch 0 tubelet 0: {rgb_masks[0, 0]}')


        for b in range(B):
            rgb_mask_index = 0
            depth_mask_index = 0
            logger.info(f'Num tubelets: {num_tubelets}')
            for tubelet in range(num_tubelets):
                
                #if the tensor at the given position is full of zeros (768), then insert the encoded tensor
                if (rgb_masks[b, tubelet] == False).all():
                    rgb_encoded_with_mask[b, tubelet] = encoded_rgb_embeddings[b, rgb_mask_index]
                    rgb_mask_index += 1
                else:
                    rgb_encoded_with_mask[b, tubelet] = mask_tokens_rgb[b, tubelet]
                
                if (depth_masks[b, tubelet] == False).all():
                    depth_encoded_with_mask[b, tubelet] = encoded_depth_embeddings[b, depth_mask_index]
                    depth_mask_index += 1
                else:
                    depth_encoded_with_mask[b, tubelet] = mask_tokens_depth[b, tubelet]
                
                    
                    

        logger.info(f'RGB Encoded with Mask shape: {rgb_encoded_with_mask.shape}')
        #logger.info(f'RGB Encoded with Mask: {rgb_encoded_with_mask}')

        logger.info(f'Depth Encoded with Mask shape: {depth_encoded_with_mask.shape}')
        #logger.info(f'Depth Encoded with Mask: {depth_encoded_with_mask}')

        #Initialize the mask tokens with nn.init.trunc_normal_(self.mask_token, std=0.02)
                # Initialize the mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim)) 
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        #For each zero tensor in the rgb_encoded_with_mask, replace it with the mask token
        for b in range(B):
            for tubelet in range(num_tubelets):
                if (rgb_encoded_with_mask[b, tubelet] == 0).all():
                    rgb_encoded_with_mask[b, tubelet] = self.mask_token
                if (depth_encoded_with_mask[b, tubelet] == 0).all():
                    depth_encoded_with_mask[b, tubelet] = self.mask_token

        logger.info(f'RGB Encoded with Mask after adding normal distribution init shape: {rgb_encoded_with_mask.shape}')
        #logger.info(f'RGB Encoded with Mask after adding normal distribution init: {rgb_encoded_with_mask}')
        logger.info(f'Depth Encoded with Mask after adding normal distribution init shape: {depth_encoded_with_mask.shape}')
        #logger.info(f'Depth Encoded with Mask after adding normal distribution init: {depth_encoded_with_mask}')


        rgb_decoder_input = rgb_encoded_with_mask
        depth_decoder_input = depth_encoded_with_mask
        logger.info(f'RGB Decoder Input shape: {rgb_decoder_input.shape}')
        logger.info(f'Depth Decoder Input shape: {depth_decoder_input.shape}')

        # Prepare the sequence for decoders
        # RGB Decoder
        rgb_decoder_input = self.encoder_to_decoder(rgb_decoder_input) if self.embed_dim != self.decoder_embed_dim else rgb_decoder_input  #REVISAR
        logger.info(f'RGB Decoder Input shape after encoder to decoder: {rgb_decoder_input.shape}')
        rgb_decoder_embeddings = rgb_decoder_input
        for block in self.rgb_decoder:
            rgb_decoder_embeddings = block(rgb_decoder_embeddings)

        rgb_decoder_embeddings_out = self.rgb_decoder_norm(rgb_decoder_embeddings)
        logger.info(f'RGB Decoder Embeddings Out shape: {rgb_decoder_embeddings_out.shape}')
        rgb_reconstruction = self.rgb_head(rgb_decoder_embeddings_out)   # Shape: [B, N_tubelets, num_pixels_per_tubelet]
        logger.info(f'RGB Reconstruction shape after head: {rgb_reconstruction.shape}')

        # Depth Decoder
        depth_decoder_input = self.encoder_to_decoder(depth_decoder_input) if self.embed_dim != self.decoder_embed_dim else depth_decoder_input 
        logger.info(f'Depth Decoder Input shape after encoder to decoder: {depth_decoder_input.shape}')
        depth_decoder_embeddings = depth_decoder_input
        for block in self.depth_decoder:
            depth_decoder_embeddings = block(depth_decoder_embeddings)

        depth_decoder_embeddings_out = self.depth_decoder_norm(depth_decoder_embeddings)
        logger.info(f'Depth Decoder Embeddings Out shape: {depth_decoder_embeddings_out.shape}')
        depth_reconstruction = self.depth_head(depth_decoder_embeddings_out)  # Shape: [B, N_tubelets, num_pixels_per_tubelet]
        logger.info(f'Depth Reconstruction shape after head: {depth_reconstruction.shape}')

        num_pixels_per_patch = self.patch_size ** 2
        #Now we need to reshape the output tensors to the desired shape. from [B, N_tubelets, num_pixels_per_tubelet] to [B, T, num_patches_per_frame, num_pixels_per_patch*channels]
        RGB_frame_reconstruction = rgb_reconstruction.view(B, T, num_patches_per_frame, num_pixels_per_patch*3)
        logger.info(f'RGB Frame Reconstruction shape: {RGB_frame_reconstruction.shape}')
        
        depth_frame_reconstruction = depth_reconstruction.view(B, T, num_patches_per_frame, num_pixels_per_patch*1)
        logger.info(f'Depth Frame Reconstruction shape: {depth_frame_reconstruction.shape}')

        logger.info(f'RGB Mask shape: {rgb_masks.shape}')   
        logger.info(f'Depth Mask shape: {depth_masks.shape}')
        #only keep first two dimensions, eliminate the third one so that we get [B, N_tubelets], for that first turn las dimension to 1
        rgb_masks = rgb_masks[:,:,0]
        depth_masks = depth_masks[:,:,0]
        

        logger.info(f'RGB Mask shape after eliminating last dim: {rgb_masks.shape}')
        logger.info(f'Depth Mask shape after eliminating last dim: {depth_masks.shape}')


        rgb_masks_reshaped = rgb_masks.view(B, num_temporal_positions, num_patches_per_frame)
        depth_masks_reshaped = depth_masks.view(B, num_temporal_positions, num_patches_per_frame)
        
        logger.info(f'RGB Masks Reshaped shape: {rgb_masks_reshaped.shape}')
        logger.info(f'Depth Masks Reshaped shape: {depth_masks_reshaped.shape}')

        # Expand along the tubelet_size frames
        rgb_masks_expanded = rgb_masks_reshaped.unsqueeze(2).expand(-1, -1, self.tubelet_size, -1)
        depth_masks_expanded = depth_masks_reshaped.unsqueeze(2).expand(-1, -1, self.tubelet_size, -1)
        logger.info(f'RGB Masks Expanded shape: {rgb_masks_expanded.shape}')
        logger.info(f'Depth Masks Expanded shape: {depth_masks_expanded.shape}')

        # Merge to get masks per frame [B, T, num_patches_per_frame]
        rgb_masks_per_frame = rgb_masks_expanded.contiguous().view(B, T, num_patches_per_frame)
        depth_masks_per_frame = depth_masks_expanded.contiguous().view(B, T, num_patches_per_frame)

        logger.info(f'RGB Masks per Frame shape: {rgb_masks_per_frame.shape}')
        logger.info(f'Depth Masks per Frame shape: {depth_masks_per_frame.shape}')

        rgb_masks_per_frame = rgb_masks_per_frame.long()
        depth_masks_per_frame = depth_masks_per_frame.long()

        logger.info(f'RGB Masks per Frame shape after long: {rgb_masks_per_frame.shape}')
        logger.info(f'Depth Masks per Frame shape after long: {depth_masks_per_frame.shape}')

        return RGB_frame_reconstruction, depth_frame_reconstruction, rgb_masks_per_frame, depth_masks_per_frame


if __name__ == "__main__":
    import argparse
    import yaml

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load configuration settings from config.yaml
    with open('../config/config.yaml', 'r') as f:
        full_config = yaml.safe_load(f)

    # Combine model and training configs
    config = full_config['model']
    training_config = full_config['training']
    data_config = full_config['data']

    # Add training parameters to config
    config['mask_ratio'] = training_config['mask_ratio']
    config['alpha'] = training_config.get('alpha', 1.0) 
    config['beta'] = training_config.get('beta', 1.0)

    # Instantiate the model
    model = CrossModalVideoMAE(config)
    logger.info("Model instantiated successfully.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Create dummy input data
    B = 2  # Batch size (set to 1 for visualization purposes)
    C_rgb = 3
    C_depth = 1
    T = config['num_frames']
    H = config['img_size']
    W = config['img_size']

    rgb_frames = torch.randn(B, C_rgb, T, H, W).to(device)
    depth_maps = torch.randn(B, C_depth, T, H, W).to(device)
    logger.info(f"Dummy RGB frames shape: {rgb_frames.shape}")
    logger.info(f"Dummy Depth maps shape: {depth_maps.shape}")

    # Forward pass
    with torch.no_grad():
        rgb_reconstruction, depth_reconstruction, rgb_masks, depth_masks = model(rgb_frames, depth_maps)
        logger.info("Forward pass completed.")

    logger.info("All assessments passed successfully.")
