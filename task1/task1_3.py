import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import os
from task1_1 import visualize_tibia_femur_slices


def load_ct_scan(file_path):
    return nib.load(file_path)


def get_structuring_element_for_mm(expansion_mm, voxel_spacing):
    radius_voxels = [int(np.ceil(expansion_mm / sp)) for sp in voxel_spacing]
    zz, yy, xx = np.ogrid[
        -radius_voxels[0]:radius_voxels[0]+1,
        -radius_voxels[1]:radius_voxels[1]+1,
        -radius_voxels[2]:radius_voxels[2]+1
    ]
    ellipsoid = ((zz * voxel_spacing[0])**2 +
                 (yy * voxel_spacing[1])**2 +
                 (xx * voxel_spacing[2])**2) <= expansion_mm**2
    return ellipsoid


def expand_mask(mask_data, expansion_mm, voxel_spacing):
    struct_elem = get_structuring_element_for_mm(expansion_mm, voxel_spacing)
    return binary_dilation(mask_data, structure=struct_elem).astype(np.uint8)


def randomize_expansion(original_mask, expanded_mask, random_fraction=0.5):
    """
    Randomly selects a subset of voxels from the expanded region not in the original mask.
    The selection is controlled by `random_fraction`.
    """
    expansion_region = (expanded_mask == 1) & (original_mask == 0)
    randomized_region = (np.random.rand(*original_mask.shape) < random_fraction) & expansion_region
    randomized_mask = original_mask | randomized_region
    return randomized_mask.astype(np.uint8)


def process_and_save_randomized(mask_path, ct_data, max_expansion_mm=2.0, random_fraction=0.5):
    img = load_ct_scan(mask_path)
    voxel_spacing = img.header.get_zooms()
    original_mask = img.get_fdata().astype(bool)

    expanded_mask = expand_mask(original_mask, max_expansion_mm, voxel_spacing)
    randomized_mask = randomize_expansion(original_mask, expanded_mask, random_fraction)

    randomized_img = nib.Nifti1Image(randomized_mask, affine=img.affine, header=img.header)
    output_path = mask_path.replace('.nii.gz', f'_randomized_{int(max_expansion_mm)}mm_frac{int(random_fraction*100)}_2nd.nii.gz')
    nib.save(randomized_img, output_path)

    print(f"Saved randomized mask to: {output_path}")

    # Optional visualization
    # if "femur" in mask_path:
    #     visualize_tibia_femur_slices(ct_data, randomized_mask, None, filename=os.path.join("results", 'femur_slices_randomized.png'))
    # else:
    #     visualize_tibia_femur_slices(ct_data, None, randomized_mask, filename=os.path.join("results", 'tibia_slices_randomized.png'))


if __name__ == "__main__":
    img = load_ct_scan("3702_left_knee.nii.gz")
    ct_data = img.get_fdata()

    process_and_save_randomized('results/original_femur_segmentation.nii.gz', ct_data=ct_data, max_expansion_mm=2.0, random_fraction=0.5)
    process_and_save_randomized('results/original_tibia_segmentation.nii.gz', ct_data=ct_data, max_expansion_mm=2.0, random_fraction=0.5)
