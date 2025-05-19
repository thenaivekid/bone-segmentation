import numpy as np
import nibabel as nib
import os
from scipy.ndimage import binary_dilation
from task1_1 import load_ct_scan



def detect_medial_lateral_lowest_points(mask_data, affine):
    coords = np.array(np.where(mask_data)).T  # [N, 3]
    world_coords = nib.affines.apply_affine(affine, coords)

    # Sort by z (inferior to superior)
    sorted_indices = np.argsort(world_coords[:, 2])
    sorted_coords = world_coords[sorted_indices]

    # Split by x (medial/lateral) axis
    x_median = np.median(world_coords[:, 0])
    medial_point = None
    lateral_point = None

    for point in sorted_coords:
        if point[0] < x_median and medial_point is None:
            medial_point = point
        elif point[0] >= x_median and lateral_point is None:
            lateral_point = point
        if medial_point is not None and lateral_point is not None:
            break

    return medial_point, lateral_point


def process_tibia_masks(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    original_mask, affine = load_ct_scan("results/original_tibia_segmentation.nii.gz")
    
    # Original
    results['original'] = detect_medial_lateral_lowest_points(original_mask, affine)

    # 2mm
    expanded_2mm, affine = load_ct_scan("/home/ashok/bone_seg/results/femur_segmentation_expanded_2mm.nii.gz")
    results['expanded_2mm'] = detect_medial_lateral_lowest_points(expanded_2mm, affine)

    # 4mm
    expanded_4mm, affine = load_ct_scan("/home/ashok/bone_seg/results/original_femur_segmentation_expanded_4mm.nii.gz")
    results['expanded_4mm'] = detect_medial_lateral_lowest_points(expanded_4mm, affine)

    # Random 1
    rand_mask_1, affine = load_ct_scan("/home/ashok/bone_seg/results/femur_segmentation_randomized_2mm_frac50.nii.gz")
    results['random_1'] = detect_medial_lateral_lowest_points(rand_mask_1, affine)

    # Random 2
    rand_mask_2, affine = load_ct_scan("/home/ashok/bone_seg/results/original_femur_segmentation_randomized_2mm_frac50_2nd.nii.gz")
    results['random_2'] = detect_medial_lateral_lowest_points(rand_mask_2, affine)

    # Print all coordinates
    for key, (medial, lateral) in results.items():
        print(f"\n== {key.upper()} ==")
        print(f"Medial: {medial}")
        print(f"Lateral: {lateral}")

    # Save to CSV (optional)
    with open(os.path.join(output_dir, "tibia_landmarks.csv"), "w") as f:
        f.write("Mask,Medial_X,Medial_Y,Medial_Z,Lateral_X,Lateral_Y,Lateral_Z\n")
        for key, (medial, lateral) in results.items():
            f.write(f"{key},{','.join(map(str, medial))},{','.join(map(str, lateral))}\n")


if __name__ == "__main__":
    process_tibia_masks(output_dir="results")
