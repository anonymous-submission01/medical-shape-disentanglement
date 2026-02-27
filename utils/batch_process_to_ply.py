#!/usr/bin/env python3
"""
ADNI Hippocampus Batch Processing to PLY Format with Volume Correction

This script processes ALL separated hippocampus NIfTI masks and converts them to PLY format
using TWO approaches, both with volume correction to maintain ground truth volumes:

1. Minimal + Volume Correction (no smoothing)
   - Binary → toMesh(0.5) → center → volume correction → global scale
   - Preserves blocky surface but accurate volume
   
2. Minimal + Smooth + Volume Correction
   - Binary → toMesh(0.5) → smooth(3) → center → volume correction → global scale
   - Smooth surface with accurate volume

Each group (left, right, combined) uses SEPARATE global scale factors:
- Left hippocampus: Global scale based on max dimension of all left hippocampi
- Right hippocampus: Global scale based on max dimension of all right hippocampi
- Combined hippocampus: Global scale based on max dimension of all combined hippocampi

Output Structure:
  adni_processed/
    left_hippocampus_ply/
      minimal/               - PLY files (minimal + volume corrected)
      minimal_smooth/        - PLY files (minimal + smooth + volume corrected)
      metadata.csv          - Global scale, volume unscale, statistics
    right_hippocampus_ply/
      minimal/
      minimal_smooth/
      metadata.csv
    combined_hippocampus_ply/
      minimal/
      minimal_smooth/
      metadata.csv

Requirements:
- Separated hippocampus NIfTI files must exist (from adni_processing.ipynb cell 8)
- ShapeWorks library for grooming pipeline
- trimesh library for volume calculations

Usage:
    python batch_process_to_ply.py
"""

import os
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from tqdm import tqdm
import shapeworks as sw
import trimesh
import tempfile
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack


@contextmanager
def suppress_stdout(out=True, err=False):
    """Suppress stdout and/or stderr output to reduce console clutter"""
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield


def get_nifti_dimension_fast(nifti_file):
    """
    Quickly get the physical dimension of a NIfTI mask without full processing.
    Just reads the data and calculates bounding box in physical coordinates.
    This is ~40x faster than doing full grooming pipeline.
    
    Args:
        nifti_file: Path to NIfTI file
    
    Returns:
        float or None: Maximum dimension in mm (max of X, Y, Z extents)
    """
    try:
        img = nib.load(nifti_file)
        data = img.get_fdata()
        coords = np.where(data > 0)
        
        if len(coords[0]) == 0:
            return None
        
        voxel_dims = img.header.get_zooms()
        
        # Calculate extents in each dimension
        max_dim = max([
            (np.max(coords[i]) - np.min(coords[i])) * voxel_dims[i] 
            for i in range(3)
        ])
        
        return max_dim
        
    except Exception as e:
        print(f"Error reading {os.path.basename(nifti_file)}: {e}")
        return None


def process_minimal_volcorrect(input_file, output_file, global_scale):
    """
    Process hippocampus with MINIMAL approach + VOLUME CORRECTION.
    
    Pipeline:
    1. Binary mask → toMesh(0.5) [marching cubes at 0.5 isovalue]
    2. Center mesh at origin
    3. Calculate ground truth volume from voxel count
    4. Measure mesh volume
    5. Apply volume correction: scale = (V_true / V_mesh)^(1/3)
    6. Apply global scale factor (for normalization)
    
    Args:
        input_file: Path to input NIfTI file
        output_file: Path to output PLY file
        global_scale: Global scale factor (same for all shapes in this group)
    
    Returns:
        tuple: (success: bool, volume_correction_factor: float)
    """
    try:
        with suppress_stdout():
            # Get ground truth volume from voxel count
            nifti_img = nib.load(input_file)
            nifti_data = nifti_img.get_fdata()
            voxel_dims = nifti_img.header.get_zooms()
            voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]  # mm³ per voxel
            voxel_count = np.sum(nifti_data > 0)
            true_volume_mm3 = voxel_count * voxel_volume
            
            # Create mesh using minimal pipeline (no smoothing)
            shape_seg = sw.Image(input_file)
            shape_seg.binarize(0)  # Binary threshold
            shape_seg.pad(5, 0)    # Padding to avoid boundary effects
            mesh_shape = shape_seg.toMesh(0.5)  # Marching cubes at isovalue 0.5
            
            # Center mesh at origin
            center = mesh_shape.center()
            mesh_shape.translate(list(-center))
            
            # Calculate mesh volume BEFORE any scaling
            # Use temporary file to get trimesh volume
            temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            mesh_shape.write(temp_path)
            temp_mesh = trimesh.load(temp_path)
            mesh_volume_mm3 = temp_mesh.volume
            os.unlink(temp_path)
            
            # VOLUME CORRECTION: Scale mesh to match ground truth volume
            # Formula: s = (V_true / V_mesh)^(1/3)
            # This preserves the shape while correcting the volume
            volume_correction = (true_volume_mm3 / mesh_volume_mm3) ** (1.0/3.0)
            mesh_shape = mesh_shape.scale([volume_correction] * 3)
            
            # Apply GLOBAL SCALE (for normalization across dataset)
            mesh_shape = mesh_shape.scale([global_scale] * 3)
            
            # Save final PLY file
            mesh_shape.write(output_file)
        
        return True, volume_correction
        
    except Exception as e:
        print(f"Error processing {os.path.basename(input_file)}: {e}")
        return False, None


def process_minimal_smooth_volcorrect(input_file, output_file, global_scale):
    """
    Process hippocampus with MINIMAL + SMOOTH approach + VOLUME CORRECTION.
    
    Pipeline:
    1. Binary mask → toMesh(0.5) [marching cubes]
    2. Smooth mesh (3 iterations) [creates smooth surface but changes volume]
    3. Center mesh at origin
    4. Calculate ground truth volume from voxel count
    5. Measure SMOOTHED mesh volume
    6. Apply volume correction: scale = (V_true / V_smoothed)^(1/3)
    7. Apply global scale factor
    
    Args:
        input_file: Path to input NIfTI file
        output_file: Path to output PLY file
        global_scale: Global scale factor (same for all shapes in this group)
    
    Returns:
        tuple: (success: bool, volume_correction_factor: float)
    """
    try:
        with suppress_stdout():
            # Get ground truth volume from voxel count
            nifti_img = nib.load(input_file)
            nifti_data = nifti_img.get_fdata()
            voxel_dims = nifti_img.header.get_zooms()
            voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
            voxel_count = np.sum(nifti_data > 0)
            true_volume_mm3 = voxel_count * voxel_volume
            
            # Create mesh and smooth
            shape_seg = sw.Image(input_file)
            shape_seg.binarize(0)
            shape_seg.pad(5, 0)
            mesh_shape = shape_seg.toMesh(0.5)
            
            # SMOOTH (this changes volume - we'll correct it later)
            mesh_shape.smooth(3, 1)  # 3 iterations, relaxation factor 1
            
            # Center mesh
            center = mesh_shape.center()
            mesh_shape.translate(list(-center))
            
            # Calculate SMOOTHED mesh volume BEFORE any scaling
            temp_file = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            mesh_shape.write(temp_path)
            temp_mesh = trimesh.load(temp_path)
            smoothed_volume_mm3 = temp_mesh.volume
            os.unlink(temp_path)
            
            # VOLUME CORRECTION: Scale to match ground truth volume
            # This maintains the smooth surface while correcting the volume
            volume_correction = (true_volume_mm3 / smoothed_volume_mm3) ** (1.0/3.0)
            mesh_shape = mesh_shape.scale([volume_correction] * 3)
            
            # Apply GLOBAL SCALE
            mesh_shape = mesh_shape.scale([global_scale] * 3)
            
            # Save final PLY file
            mesh_shape.write(output_file)
        
        return True, volume_correction
        
    except Exception as e:
        print(f"Error processing {os.path.basename(input_file)}: {e}")
        return False, None


def process_group(input_files, output_base_dir, group_name):
    """
    Process a group of hippocampus files (left, right, or combined) with both approaches.
    
    This function:
    1. Calculates global scale factor for this group
    2. Processes all files with minimal + volume correction
    3. Processes all files with minimal + smooth + volume correction
    4. Saves metadata (global scale, volume unscale factors, statistics)
    
    Args:
        input_files: List of input NIfTI file paths
        output_base_dir: Base output directory for this group
        group_name: Name of group ("left", "right", or "combined")
    
    Returns:
        dict: Processing statistics and metadata
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING {group_name.upper()} HIPPOCAMPUS")
    print(f"{'='*80}")
    print(f"Total files: {len(input_files)}")
    
    # Create output directories
    minimal_dir = os.path.join(output_base_dir, "minimal")
    smooth_dir = os.path.join(output_base_dir, "minimal_smooth")
    os.makedirs(minimal_dir, exist_ok=True)
    os.makedirs(smooth_dir, exist_ok=True)
    
    # ============================================================================
    # STEP 1: Calculate global scale factor for this group
    # ============================================================================
    print(f"\nStep 1: Calculating global scale factor...")
    
    max_dimensions = []
    for input_file in tqdm(input_files, desc="Finding max dimension"):
        max_dim = get_nifti_dimension_fast(input_file)
        if max_dim is not None:
            max_dimensions.append(max_dim)
    
    if len(max_dimensions) == 0:
        print(f"❌ Error: Could not calculate dimensions for any files!")
        return None
    
    global_max_dimension = max(max_dimensions)
    min_dimension = min(max_dimensions)
    mean_dimension = np.mean(max_dimensions)
    std_dimension = np.std(max_dimensions)
    
    # Apply safety buffer (1.2x) to ensure largest shape fits after padding/smoothing
    global_max_dimension_buffered = global_max_dimension * 1.2
    global_scale = 1.0 / global_max_dimension_buffered
    
    # Volume unscale factor: to recover original mm³ from PLY volume
    # Formula: original_volume = ply_volume * (1/global_scale)³
    volume_unscale = (1.0 / global_scale) ** 3
    
    print(f"\n✓ Global scale calculation complete:")
    print(f"  Dimension range: {min_dimension:.2f} - {global_max_dimension:.2f} mm")
    print(f"  Mean dimension: {mean_dimension:.2f} ± {std_dimension:.2f} mm")
    print(f"  Max dimension (buffered): {global_max_dimension_buffered:.2f} mm")
    print(f"  Global scale factor: {global_scale:.8f}")
    print(f"  Volume unscale factor: {volume_unscale:.6e}")
    print(f"\nTo recover original measurements from PLY files:")
    print(f"  • Distance (mm) = PLY_distance × {1/global_scale:.2f}")
    print(f"  • Volume (mm³) = PLY_volume × {volume_unscale:.6e}")
    
    # ============================================================================
    # STEP 2: Process with MINIMAL + Volume Correction
    # ============================================================================
    print(f"\nStep 2: Processing with MINIMAL + Volume Correction...")
    
    minimal_success = 0
    minimal_failed = 0
    minimal_vol_corrections = []
    
    for input_file in tqdm(input_files, desc="Minimal processing"):
        basename = os.path.basename(input_file).replace('.nii.gz', '').replace('.nii', '')
        output_file = os.path.join(minimal_dir, f"{basename}.ply")
        
        success, vol_corr = process_minimal_volcorrect(input_file, output_file, global_scale)
        if success:
            minimal_success += 1
            if vol_corr is not None:
                minimal_vol_corrections.append(vol_corr)
        else:
            minimal_failed += 1
    
    print(f"\n✓ Minimal processing complete:")
    print(f"  Success: {minimal_success}/{len(input_files)}")
    print(f"  Failed: {minimal_failed}")
    if len(minimal_vol_corrections) > 0:
        print(f"  Volume correction factors: {np.mean(minimal_vol_corrections):.4f} ± {np.std(minimal_vol_corrections):.4f}")
    
    # ============================================================================
    # STEP 3: Process with MINIMAL + SMOOTH + Volume Correction
    # ============================================================================
    print(f"\nStep 3: Processing with MINIMAL + SMOOTH + Volume Correction...")
    
    smooth_success = 0
    smooth_failed = 0
    smooth_vol_corrections = []
    
    for input_file in tqdm(input_files, desc="Smooth processing"):
        basename = os.path.basename(input_file).replace('.nii.gz', '').replace('.nii', '')
        output_file = os.path.join(smooth_dir, f"{basename}.ply")
        
        success, vol_corr = process_minimal_smooth_volcorrect(input_file, output_file, global_scale)
        if success:
            smooth_success += 1
            if vol_corr is not None:
                smooth_vol_corrections.append(vol_corr)
        else:
            smooth_failed += 1
    
    print(f"\n✓ Smooth processing complete:")
    print(f"  Success: {smooth_success}/{len(input_files)}")
    print(f"  Failed: {smooth_failed}")
    if len(smooth_vol_corrections) > 0:
        print(f"  Volume correction factors: {np.mean(smooth_vol_corrections):.4f} ± {np.std(smooth_vol_corrections):.4f}")
    
    # ============================================================================
    # STEP 4: Save metadata
    # ============================================================================
    print(f"\nStep 4: Saving metadata...")
    
    metadata = {
        'group': [group_name],
        'total_files': [len(input_files)],
        'minimal_success': [minimal_success],
        'minimal_failed': [minimal_failed],
        'smooth_success': [smooth_success],
        'smooth_failed': [smooth_failed],
        'dimension_min_mm': [min_dimension],
        'dimension_max_mm': [global_max_dimension],
        'dimension_mean_mm': [mean_dimension],
        'dimension_std_mm': [std_dimension],
        'dimension_max_buffered_mm': [global_max_dimension_buffered],
        'global_scale_factor': [global_scale],
        'distance_unscale_factor': [1.0 / global_scale],
        'volume_unscale_factor': [volume_unscale],
        'minimal_vol_corr_mean': [np.mean(minimal_vol_corrections) if len(minimal_vol_corrections) > 0 else None],
        'minimal_vol_corr_std': [np.std(minimal_vol_corrections) if len(minimal_vol_corrections) > 0 else None],
        'smooth_vol_corr_mean': [np.mean(smooth_vol_corrections) if len(smooth_vol_corrections) > 0 else None],
        'smooth_vol_corr_std': [np.std(smooth_vol_corrections) if len(smooth_vol_corrections) > 0 else None],
    }
    
    metadata_df = pd.DataFrame(metadata)
    metadata_csv = os.path.join(output_base_dir, "metadata.csv")
    metadata_df.to_csv(metadata_csv, index=False)
    
    print(f"✓ Metadata saved to: {metadata_csv}")
    
    # Also save a detailed README
    readme_path = os.path.join(output_base_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write(f"{group_name.upper()} HIPPOCAMPUS PLY FILES - METADATA\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total files processed: {len(input_files)}\n")
        f.write(f"  Minimal + VolCorr: {minimal_success} success, {minimal_failed} failed\n")
        f.write(f"  Smooth + VolCorr: {smooth_success} success, {smooth_failed} failed\n\n")
        f.write(f"GLOBAL SCALE INFORMATION\n")
        f.write(f"{'='*80}\n")
        f.write(f"Dimension range: {min_dimension:.2f} - {global_max_dimension:.2f} mm\n")
        f.write(f"Mean dimension: {mean_dimension:.2f} ± {std_dimension:.2f} mm\n")
        f.write(f"Max dimension (with buffer): {global_max_dimension_buffered:.2f} mm\n")
        f.write(f"Global scale factor: {global_scale:.8f}\n\n")
        f.write(f"RECOVERING ORIGINAL MEASUREMENTS\n")
        f.write(f"{'='*80}\n")
        f.write(f"To recover original measurements from PLY files:\n\n")
        f.write(f"1. Distance (mm):\n")
        f.write(f"   original_distance_mm = ply_distance × {1/global_scale:.6f}\n\n")
        f.write(f"2. Volume (mm³):\n")
        f.write(f"   original_volume_mm3 = ply_volume × {volume_unscale:.6e}\n\n")
        f.write(f"VOLUME CORRECTION\n")
        f.write(f"{'='*80}\n")
        f.write(f"Both approaches use volume correction to maintain ground truth volumes:\n")
        f.write(f"  • Minimal: Volume correction factor = {np.mean(minimal_vol_corrections):.4f} ± {np.std(minimal_vol_corrections):.4f}\n")
        f.write(f"  • Smooth: Volume correction factor = {np.mean(smooth_vol_corrections):.4f} ± {np.std(smooth_vol_corrections):.4f}\n\n")
        f.write(f"This ensures that volumes calculated from PLY files match ground truth\n")
        f.write(f"when unscaled using the volume_unscale_factor.\n\n")
        f.write(f"DIRECTORY STRUCTURE\n")
        f.write(f"{'='*80}\n")
        f.write(f"minimal/           - PLY files with minimal processing + volume correction\n")
        f.write(f"                     (blocky surface, accurate volume)\n\n")
        f.write(f"minimal_smooth/    - PLY files with smoothing + volume correction\n")
        f.write(f"                     (smooth surface, accurate volume)\n\n")
        f.write(f"metadata.csv       - Machine-readable metadata\n")
        f.write(f"README.txt         - This file (human-readable description)\n")
    
    print(f"✓ README saved to: {readme_path}")
    
    # Return summary
    return {
        'group': group_name,
        'total_files': len(input_files),
        'minimal_success': minimal_success,
        'smooth_success': smooth_success,
        'global_scale': global_scale,
        'volume_unscale': volume_unscale,
        'minimal_dir': minimal_dir,
        'smooth_dir': smooth_dir
    }


def main():
    """Main batch processing function"""
    
    print("="*80)
    print("ADNI HIPPOCAMPUS BATCH PROCESSING TO PLY")
    print("Volume-Corrected Minimal and Minimal+Smooth Versions")
    print("="*80)
    
    # Set paths
    output_dir = "[PATH_TO_ADNI_PROCESSED]"  # TODO: Update this to your actual path
    
    # Input directories (separated NIfTI files)
    left_hippo_dir = os.path.join(output_dir, "left_hippocampus_Mask")
    right_hippo_dir = os.path.join(output_dir, "right_hippocampus_Mask")
    combined_hippo_dir = os.path.join(output_dir, "combined_hippocampus_Mask")
    
    # Output directories (PLY files with subdirectories for each approach)
    left_ply_base = os.path.join(output_dir, "left_hippocampus_ply")
    right_ply_base = os.path.join(output_dir, "right_hippocampus_ply")
    combined_ply_base = os.path.join(output_dir, "combined_hippocampus_ply")
    
    os.makedirs(left_ply_base, exist_ok=True)
    os.makedirs(right_ply_base, exist_ok=True)
    os.makedirs(combined_ply_base, exist_ok=True)
    
    # Get all separated NIfTI files
    left_files = sorted(glob(os.path.join(left_hippo_dir, "*.nii*")))
    right_files = sorted(glob(os.path.join(right_hippo_dir, "*.nii*")))
    combined_files = sorted(glob(os.path.join(combined_hippo_dir, "*.nii*")))
    
    print(f"\nInput files found:")
    print(f"  Left hippocampus: {len(left_files)} files")
    print(f"  Right hippocampus: {len(right_files)} files")
    print(f"  Combined hippocampus: {len(combined_files)} files")
    print(f"  Total: {len(left_files) + len(right_files) + len(combined_files)} files")
    
    if len(left_files) == 0 and len(right_files) == 0 and len(combined_files) == 0:
        print("\n❌ ERROR: No separated hippocampus files found!")
        print("Please run the separation step first (cell 8 in adni_processing.ipynb)")
        return
    
    print(f"\nOutput directories:")
    print(f"  Left PLY: {left_ply_base}")
    print(f"  Right PLY: {right_ply_base}")
    print(f"  Combined PLY: {combined_ply_base}")
    
    # Process each group
    results = []
    
    # Process LEFT hippocampus
    if len(left_files) > 0:
        left_result = process_group(left_files, left_ply_base, "left")
        if left_result:
            results.append(left_result)
    
    # Process RIGHT hippocampus
    if len(right_files) > 0:
        right_result = process_group(right_files, right_ply_base, "right")
        if right_result:
            results.append(right_result)
    
    # Process COMBINED hippocampus
    if len(combined_files) > 0:
        combined_result = process_group(combined_files, combined_ply_base, "combined")
        if combined_result:
            results.append(combined_result)
    
    # ============================================================================
    # Final Summary
    # ============================================================================
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE - SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n{result['group'].upper()} HIPPOCAMPUS:")
        print(f"  Files processed: {result['total_files']}")
        print(f"  Minimal success: {result['minimal_success']}")
        print(f"  Smooth success: {result['smooth_success']}")
        print(f"  Global scale: {result['global_scale']:.8f}")
        print(f"  Volume unscale: {result['volume_unscale']:.6e}")
        print(f"  Output directories:")
        print(f"    Minimal: {result['minimal_dir']}")
        print(f"    Smooth: {result['smooth_dir']}")
    
    print(f"\n{'='*80}")
    print("KEY POINTS")
    print(f"{'='*80}")
    print(f"✓ Each group (left, right, combined) has SEPARATE global scale factors")
    print(f"✓ Two versions created for each group:")
    print(f"    • minimal/         - Blocky surface, volume corrected")
    print(f"    • minimal_smooth/  - Smooth surface, volume corrected")
    print(f"✓ Volume correction maintains ground truth volumes")
    print(f"✓ Metadata saved in each group's directory (metadata.csv, README.txt)")
    print(f"✓ Use metadata.csv to recover original mm distances and volumes")
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Check metadata.csv in each output directory for scale factors")
    print(f"2. Choose version based on your needs:")
    print(f"   • Minimal: Better for preserving exact voxel boundaries")
    print(f"   • Smooth: Better for visualization and distance calculations")
    print(f"3. Use for Chamfer distance, Earth Mover distance, or other analyses")
    print(f"4. Validate volumes against XML ground truth if needed")


if __name__ == "__main__":
    main()
