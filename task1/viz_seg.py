import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to

def load_nifti_img(path):
    return nib.load(path)

def resample_mask_to_ct(ct_img, mask_img):
    """Resample the segmentation mask to match CT scan shape and orientation"""
    return resample_from_to(mask_img, ct_img, order=0)  # nearest neighbor interpolation

def show_overlay_resampled(ct_img, mask_img, slice_index=108, alpha=0.4):
    """Overlay after resampling the mask"""
    ct_data = ct_img.get_fdata()
    mask_data = mask_img.get_fdata()

    if slice_index is None:
        slice_index = ct_data.shape[1] // 2

    ct_slice = ct_data[:, :, slice_index]
    mask_slice = mask_data[:, :, slice_index]

    plt.figure(figsize=(8, 8))
    plt.imshow(ct_slice.T, cmap='gray', origin='lower')
    plt.imshow(mask_slice.T, cmap='Reds', alpha=alpha, origin='lower')
    plt.title(f' Overlay (Slice y={slice_index})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    ct_img = load_nifti_img('3702_left_knee.nii.gz')
    # mask_img = load_nifti_img('results/femur_segmentation.nii.gz')
    mask_img = load_nifti_img('/home/ashok/bone_seg/results/femur_segmentation_randomized_2mm_frac50.nii.gz')

    # print(f"ct_image.header.get_zooms()=")
    # Align the mask to CT
    mask_resampled = resample_mask_to_ct(ct_img, mask_img)

    show_overlay_resampled(ct_img, mask_resampled)
