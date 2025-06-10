import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology, segmentation
from scipy import ndimage
import os
import time
from skimage.feature import peak_local_max

def load_ct_scan(file_path):
    ct_img = nib.load(file_path)
    ct_data = ct_img.get_fdata()
    return ct_data, ct_img.affine


def save_segmentation(segmentation, affine, output_file):
    seg_img = nib.Nifti1Image(segmentation.astype(np.int16), affine)
    nib.save(seg_img, output_file)

def watershed_bone_segmentation(ct_data, mask=None):
    ct_norm = (ct_data - np.min(ct_data)) / (np.max(ct_data) - np.min(ct_data))

    if mask is None:
        threshold = filters.threshold_otsu(ct_norm[ct_norm > 0.1])
        mask = ct_norm > threshold
        mask = morphology.remove_small_objects(mask, min_size=100)

    mask = morphology.binary_closing(mask, morphology.ball(2))
    mask = morphology.binary_dilation(mask, morphology.ball(1))

    distance = ndimage.distance_transform_edt(mask)
    local_max = peak_local_max(distance, min_distance=10, labels=mask)
    markers = np.zeros_like(distance, dtype=np.int32)
    markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
    watershed_result = segmentation.watershed(-distance, markers, mask=mask)

    return watershed_result, mask

def identify_tibia_femur(labeled_mask):
    properties = measure.regionprops(labeled_mask)
    volumes = [region.area for region in properties]
    centroids = [region.centroid for region in properties]
    sorted_regions = sorted(range(len(volumes)), key=lambda i: volumes[i], reverse=True)
    largest_regions = [sorted_regions[0], sorted_regions[1]]
    centroids_largest = [centroids[i] for i in largest_regions]

    if centroids_largest[0][2] < centroids_largest[1][2]:
        femur_label, tibia_label = largest_regions[0] + 1, largest_regions[1] + 1
    else:
        femur_label, tibia_label = largest_regions[1] + 1, largest_regions[0] + 1

    femur_mask = labeled_mask == femur_label
    tibia_mask = labeled_mask == tibia_label

    return femur_mask, tibia_mask

def create_background_mask(bone_mask):
    """Create background mask by inverting the bone mask"""
    background_mask = ~bone_mask
    return background_mask

def create_3d_visualization(ct_data, femur_mask, tibia_mask, background_mask=None, downsample=4, filename='bone_3d_visualization.png'):
    ct_small = ct_data[::downsample, ::downsample, ::downsample]
    femur_small = femur_mask[::downsample, ::downsample, ::downsample]
    tibia_small = tibia_mask[::downsample, ::downsample, ::downsample]

    femur_coords = np.where(femur_small)
    tibia_coords = np.where(tibia_small)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot background if provided (sample it to avoid overcrowding)
    if background_mask is not None:
        background_small = background_mask[::downsample, ::downsample, ::downsample]
        background_coords = np.where(background_small)
        # Sample background points to reduce density
        sample_indices = np.random.choice(len(background_coords[0]), 
                                        size=min(5000, len(background_coords[0])), 
                                        replace=False)
        ax.scatter(background_coords[0][sample_indices], 
                  background_coords[1][sample_indices], 
                  background_coords[2][sample_indices], 
                  c='lightgray', marker='.', s=0.5, label='Background', alpha=0.3)
    
    ax.scatter(femur_coords[0], femur_coords[1], femur_coords[2], c='red', marker='.', s=1, label='Femur', alpha=0.7)
    ax.scatter(tibia_coords[0], tibia_coords[1], tibia_coords[2], c='blue', marker='.', s=1, label='Tibia', alpha=0.7)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Visualization of Tibia, Femur, and Background')
    ax.legend()
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

def visualize_background_only(ct_data, background_mask, num_slices=6, filename='background_only_slices.png'):
    """Visualize only the background region"""
    z_indices = np.linspace(0, ct_data.shape[2]-1, num_slices).astype(int)
    fig, axes = plt.subplots(2, num_slices, figsize=(20, 8))

    for i, z in enumerate(z_indices):
        # Original CT slice
        axes[0, i].imshow(ct_data[:, :, z].T, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original CT (z={z})')

        # Background only overlay
        overlay = np.zeros((ct_data.shape[1], ct_data.shape[0], 4))
        background_slice = background_mask[:, :, z].T
        overlay[background_slice] = [0, 1, 0, 0.6]  # Green for background

        axes[1, i].imshow(ct_data[:, :, z].T, cmap='gray')
        axes[1, i].imshow(overlay)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Background Only (z={z})')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

def create_3d_background_visualization(ct_data, background_mask, downsample=4, filename='background_3d_visualization.png'):
    """Create 3D visualization of background region only"""
    background_small = background_mask[::downsample, ::downsample, ::downsample]
    background_coords = np.where(background_small)
    
    # Sample background points to make visualization manageable
    if len(background_coords[0]) > 10000:
        sample_indices = np.random.choice(len(background_coords[0]), 
                                        size=10000, 
                                        replace=False)
        background_coords = (background_coords[0][sample_indices],
                           background_coords[1][sample_indices], 
                           background_coords[2][sample_indices])

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(background_coords[0], background_coords[1], background_coords[2], 
              c='green', marker='.', s=2, label='Background', alpha=0.6)
    
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Visualization of Background Region Only')
    ax.legend()
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

def visualize_tibia_femur_slices(ct_data, femur_mask, tibia_mask, background_mask=None, num_slices=6, filename='bone_segmentation_slices.png'):
    z_indices = np.linspace(0, ct_data.shape[2]-1, num_slices).astype(int)
    fig, axes = plt.subplots(2, num_slices, figsize=(20, 8))

    for i, z in enumerate(z_indices):
        axes[0, i].imshow(ct_data[:, :, z].T, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title(f'Original CT (z={z})')

        overlay = np.zeros((ct_data.shape[1], ct_data.shape[0], 4))

        # Add background as light gray
        if background_mask is not None:
            background_slice = background_mask[:, :, z].T
            overlay[background_slice] = [0.5, 0.5, 0.5, 0.3]

        if femur_mask is not None:
            femur_slice = femur_mask[:, :, z].T
            overlay[femur_slice] = [1, 0, 0, 0.5]

        if tibia_mask is not None:
            tibia_slice = tibia_mask[:, :, z].T
            overlay[tibia_slice] = [0, 0, 1, 0.5]

        axes[1, i].imshow(ct_data[:, :, z].T, cmap='gray')
        axes[1, i].imshow(overlay)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'Segmentation (z={z})')

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()

def bone_segmentation(file_path, output_dir='results'):
    os.makedirs(output_dir, exist_ok=True)
    ct_data, affine = load_ct_scan(file_path)
    start = time.time()
    watershed_result, bone_mask = watershed_bone_segmentation(ct_data)
    elapsed = time.time() - start
    
    # Create background mask
    background_mask = create_background_mask(bone_mask)
    
    tibia_mask, femur_mask = identify_tibia_femur(watershed_result)
    
    # Save all segmentations
    save_segmentation(femur_mask, affine, os.path.join(output_dir, 'femur_segmentation.nii.gz'))
    save_segmentation(tibia_mask, affine, os.path.join(output_dir, 'tibia_segmentation.nii.gz'))
    save_segmentation(background_mask, affine, os.path.join(output_dir, 'background_segmentation.nii.gz'))
    save_segmentation(bone_mask, affine, os.path.join(output_dir, 'bone_mask.nii.gz'))
    
    # Create visualizations including background
    visualize_tibia_femur_slices(ct_data, femur_mask, tibia_mask, background_mask, 
                                filename=os.path.join(output_dir, 'tibia_femur_background_slices.png'))
    create_3d_visualization(ct_data, femur_mask, tibia_mask, background_mask, 
                           filename=os.path.join(output_dir, '3d_tibia_femur_background.png'))
    
    # Create background-only visualizations
    visualize_background_only(ct_data, background_mask, 
                             filename=os.path.join(output_dir, 'background_only_slices.png'))
    create_3d_background_visualization(ct_data, background_mask, 
                                     filename=os.path.join(output_dir, '3d_background_only.png'))
    
    print(f"Segmentation completed in {elapsed:.2f} seconds")
    print(f"Background region volume: {np.sum(background_mask)} voxels")
    print(f"Bone region volume: {np.sum(bone_mask)} voxels")
    print(f"Femur volume: {np.sum(femur_mask)} voxels")
    print(f"Tibia volume: {np.sum(tibia_mask)} voxels")
    
    return ct_data, watershed_result, femur_mask, tibia_mask, background_mask

if __name__ == "__main__":
    img_path = '3702_left_knee.nii.gz'
    ct_data, labeled_mask, femur_mask, tibia_mask, background_mask = bone_segmentation(img_path)

