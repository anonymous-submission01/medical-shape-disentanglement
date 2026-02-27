"""
Local Shapes Decoder with ReLU-based DeepSDF backbone
Implements Deep Local Shapes approach with:
- 8x8x8 grid of local latent codes (512 patches)
- 32-dim local codes per patch
- Trilinear interpolation between neighboring codes
- ReLU-based decoder network (following the original paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LocalShapesDecoder(nn.Module):
    """
    Local Shapes decoder with spatial grid of latent codes.
    Each query point gets its latent code via trilinear interpolation.
    """
    def __init__(
        self,
        latent_size,  # This will be the local code size (e.g., 16)
        dims: list,
        grid_size: int = 8,  # 8x8x8 grid
        global_latent_size: int = 256,  # Global code size for disentanglement
        encoding_features: int = 1,
        encoding_sigma: float = 0,
        xyz_in: list = (),
        xyz_in_all: bool = False,
        **siren_decoder_kwargs,
    ):
        super(LocalShapesDecoder, self).__init__()
        
        self.latent_size = latent_size  # Local code dimension (e.g., 16)
        self.global_latent_size = global_latent_size  # Global code dimension (e.g., 256)
        self.grid_size = grid_size  # 8x8x8
        self.num_local_codes = grid_size ** 3  # 512
        
        # No encoding - keep it simple like DeepSDF
        self.encoding_features = 1
        self.encoding = None
        
        # Setup xyz input dimensions for each layer
        xyz_in = list(xyz_in)
        self.latent_in = siren_decoder_kwargs.get('latent_in', [])
        self.xyz_in_all = xyz_in_all
        
        # Combined latent size: global + local
        combined_latent_size = global_latent_size + latent_size
        
        # Build ReLU-based decoder following DeepSDF
        from networks.deep_sdf_decoder import Decoder as DeepSDFDecoder
        self.decoder = DeepSDFDecoder(
            latent_size=combined_latent_size,  # Combined global + local codes
            dims=dims,
            dropout=siren_decoder_kwargs.get('dropout', None),
            dropout_prob=siren_decoder_kwargs.get('dropout_prob', 0.0),
            norm_layers=siren_decoder_kwargs.get('norm_layers', ()),
            latent_in=self.latent_in,
            weight_norm=siren_decoder_kwargs.get('weight_norm', False),
            xyz_in_all=xyz_in_all,
            use_tanh=siren_decoder_kwargs.get('use_tanh', False),
            latent_dropout=siren_decoder_kwargs.get('latent_dropout', False),
        )
    
    def get_local_codes_for_shape(self, shape_idx, all_local_codes):
        """
        Extract the grid of local codes for a specific shape.
        
        Args:
            shape_idx: Index of the shape (batch element)
            all_local_codes: [num_shapes, num_local_codes, latent_size]
        
        Returns:
            [grid_size, grid_size, grid_size, latent_size]
        """
        # Get codes for this shape: [num_local_codes, latent_size]
        shape_codes = all_local_codes[shape_idx]
        
        # Reshape to 3D grid: [grid_size, grid_size, grid_size, latent_size]
        grid_codes = shape_codes.view(self.grid_size, self.grid_size, self.grid_size, self.latent_size)
        
        return grid_codes
    
    def trilinear_interpolate(self, xyz, grid_codes, return_touched_indices=False):
        """
        Interpolate local codes at query positions using trilinear interpolation.
        
        This follows the Deep Local Shapes paper approach:
        - Divide space into regular grid of voxels
        - Each voxel corner has a latent code
        - Query point gets code via trilinear interpolation of 8 neighbors
        
        Args:
            xyz: [N, 3] query positions in [-1, 1]^3
            grid_codes: [grid_size, grid_size, grid_size, latent_size]
            return_touched_indices: If True, also return set of touched grid cell indices
        
        Returns:
            [N, latent_size] interpolated codes
            (optional) set of touched linear indices if return_touched_indices=True
        """
        N = xyz.shape[0]
        
        # Convert xyz from [-1, 1] to grid coordinates [0, grid_size-1]
        # Paper: "we discretize the volume into an N×N×N grid"
        grid_coords = (xyz + 1.0) * (self.grid_size - 1) / 2.0
        
        # Get integer parts (floor) and fractional parts for interpolation weights
        grid_coords_floor = torch.floor(grid_coords).long()
        grid_coords_frac = grid_coords - grid_coords_floor.float()
        
        # Clamp to valid range [0, grid_size-2] for floor (so ceil is at most grid_size-1)
        # This ensures we always have 8 valid neighbors for interpolation
        grid_coords_floor = torch.clamp(grid_coords_floor, 0, self.grid_size - 2)
        grid_coords_ceil = grid_coords_floor + 1
        
        # Move indices and fractional coordinates to same device as grid_codes
        device = grid_codes.device
        grid_coords_floor = grid_coords_floor.to(device)
        grid_coords_ceil = grid_coords_ceil.to(device)
        grid_coords_frac = grid_coords_frac.to(device)
        
        # Extract coordinates
        x0, y0, z0 = grid_coords_floor[:, 0], grid_coords_floor[:, 1], grid_coords_floor[:, 2]
        x1, y1, z1 = grid_coords_ceil[:, 0], grid_coords_ceil[:, 1], grid_coords_ceil[:, 2]
        
        # Get fractional parts for interpolation
        xd = grid_coords_frac[:, 0:1]  # [N, 1]
        yd = grid_coords_frac[:, 1:2]
        zd = grid_coords_frac[:, 2:3]
        
        # Get the 8 corner codes
        c000 = grid_codes[x0, y0, z0]  # [N, latent_size]
        c001 = grid_codes[x0, y0, z1]
        c010 = grid_codes[x0, y1, z0]
        c011 = grid_codes[x0, y1, z1]
        c100 = grid_codes[x1, y0, z0]
        c101 = grid_codes[x1, y0, z1]
        c110 = grid_codes[x1, y1, z0]
        c111 = grid_codes[x1, y1, z1]
        
        # Trilinear interpolation
        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd
        
        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd
        
        c = c0 * (1 - zd) + c1 * zd
        
        if return_touched_indices:
            # Convert 3D indices to linear indices for the touched cells
            # All 8 corners of each cell are accessed during interpolation
            touched_indices = set()
            for xi, yi, zi in [(x0, y0, z0), (x0, y0, z1), (x0, y1, z0), (x0, y1, z1),
                               (x1, y0, z0), (x1, y0, z1), (x1, y1, z0), (x1, y1, z1)]:
                linear_idx = (xi * self.grid_size * self.grid_size + yi * self.grid_size + zi).cpu().numpy()
                touched_indices.update(linear_idx.tolist())
            return c, touched_indices
        
        return c  # [N, latent_size]
    
    def forward(self, input_x, global_codes, all_local_codes, indices, return_touched_indices=False):
        """
        Forward pass with BOTH global and local latent codes.
        
        Args:
            input_x: [N, 3] xyz coordinates
            global_codes: [N, global_latent_size] global codes (one per point)
            all_local_codes: [num_shapes, num_local_codes, latent_size] all local codes
            indices: [N] shape indices for each query point
            return_touched_indices: If True, return dict mapping shape_idx to set of touched cell indices
        
        Returns:
            [N, 1] SDF predictions
            (optional) dict of touched indices per shape if return_touched_indices=True
        """
        xyz = input_x  # [N, 3]
        N = xyz.shape[0]
        
        # Ensure indices are on the same device as xyz
        indices = indices.to(xyz.device)
        
        # Get unique shape indices in this batch
        unique_indices = torch.unique(indices)
        
        # Storage for interpolated LOCAL codes
        interpolated_local_codes = torch.zeros(N, self.latent_size, device=xyz.device)
        
        # Track touched indices per shape (for sparse regularization)
        touched_indices_per_shape = {} if return_touched_indices else None
        
        # For each shape in the batch, interpolate LOCAL codes for its points
        for shape_idx in unique_indices:
            # Find which points belong to this shape
            mask = (indices == shape_idx)
            
            # Get grid of codes for this shape
            grid_codes = self.get_local_codes_for_shape(shape_idx, all_local_codes)
            
            # Get xyz for this shape
            shape_xyz = xyz[mask]
            
            # Interpolate local codes (optionally track touched cells)
            if return_touched_indices:
                shape_codes, touched_indices = self.trilinear_interpolate(
                    shape_xyz, grid_codes, return_touched_indices=True
                )
                touched_indices_per_shape[shape_idx.item()] = touched_indices
            else:
                shape_codes = self.trilinear_interpolate(shape_xyz, grid_codes)
            
            # Store in output
            interpolated_local_codes[mask] = shape_codes
        
        # Combine GLOBAL and LOCAL codes
        # global_codes: [N, global_latent_size] - already provided per point
        # interpolated_local_codes: [N, latent_size] - computed via interpolation
        combined_codes = torch.cat([global_codes, interpolated_local_codes], dim=1)  # [N, global + local]
        
        # Concatenate combined codes with xyz (DeepSDF format)
        # DeepSDF expects: [latent, xyz] concatenated
        decoder_input = torch.cat([combined_codes, xyz], dim=1)
        
        # Decode SDF
        sdf_pred = self.decoder(decoder_input)
        
        if return_touched_indices:
            return sdf_pred, touched_indices_per_shape
        return sdf_pred
    
    def to(self, *args, **kwargs):
        module = super(LocalShapesDecoder, self).to(*args, **kwargs)
        self.decoder = self.decoder.to(*args, **kwargs)
        return module


class Decoder(nn.Module):
    """
    Wrapper class for compatibility with existing training code.
    Supports BOTH global and local latent codes.
    """
    def __init__(
        self,
        latent_size,
        dims: list,
        grid_size: int = 8,
        global_latent_size: int = 256,
        encoding_features: int = 1,
        encoding_sigma: float = 0,
        xyz_in: list = (),
        xyz_in_all: bool = False,
        **siren_decoder_kwargs,
    ):
        super(Decoder, self).__init__()
        
        self.grid_size = grid_size
        self.num_local_codes = grid_size ** 3
        self.local_code_size = latent_size
        self.global_code_size = global_latent_size
        
        self.local_decoder = LocalShapesDecoder(
            latent_size=latent_size,
            dims=dims,
            grid_size=grid_size,
            global_latent_size=global_latent_size,
            encoding_features=encoding_features,
            encoding_sigma=encoding_sigma,
            xyz_in=xyz_in,
            xyz_in_all=xyz_in_all,
            **siren_decoder_kwargs,
        )
    
    def forward(self, input_x, global_codes, all_local_codes, indices, return_touched_indices=False):
        """
        Forward compatible with training loop.
        
        Args:
            input_x: [N, 3] xyz only (not concatenated with latent)
            global_codes: [N, global_latent_size] global codes for each point
            all_local_codes: [num_shapes, num_local_codes, local_code_size]
            indices: [N] shape indices
            return_touched_indices: If True, also return touched cell indices for sparse regularization
        
        Returns:
            [N, 1] SDF predictions
            (optional) dict of touched indices if return_touched_indices=True
        """
        return self.local_decoder(input_x, global_codes, all_local_codes, indices, return_touched_indices)
    
    def to(self, *args, **kwargs):
        module = super(Decoder, self).to(*args, **kwargs)
        self.local_decoder = self.local_decoder.to(*args, **kwargs)
        return module
