import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation
import os
from task1_1 import visualize_tibia_femur_slices

def load_ct_scan(file_path):
    return nib.load(file_path)


def get_structuring_element_for_mm(expansion_mm, voxel_spacing):
    # Convert mm to voxel units
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

def process_and_save(mask_path, ct_data, expansion_mm=2.0):
    img = load_ct_scan(mask_path)
    voxel_spacing = img.header.get_zooms()
    mask_data = img.get_fdata().astype(bool)

    expanded_mask = expand_mask(mask_data, expansion_mm, voxel_spacing)
    
    expanded_img = nib.Nifti1Image(expanded_mask.astype(np.uint8), affine=img.affine, header=img.header)
    
    output_path = mask_path.replace('.nii.gz', f'_expanded_{int(expansion_mm)}mm.nii.gz')
    nib.save(expanded_img, output_path)
    
    print(f"Saved dilated mask to: {output_path}")
    # if "femur" in mask_data:
    #     visualize_tibia_femur_slices(ct_data, expand_mask, None, filename=os.path.join("results", 'femur_slices_expanded_2.png'))
    # else:
    #     visualize_tibia_femur_slices(ct_data, None, expanded_mask, filename=os.path.join("results", 'tibia_slices_expanded_2.png'))



# Run for both femur and tibia
if __name__ == "__main__":
    img = load_ct_scan("3702_left_knee.nii.gz")
    ct_data = img.get_fdata()
    print(f"{img.header.get_zooms()=}")
    process_and_save('results/original_femur_segmentation.nii.gz', expansion_mm=4.0, ct_data = ct_data)
    process_and_save('results/original_tibia_segmentation.nii.gz', expansion_mm=4.0, ct_data = ct_data)
