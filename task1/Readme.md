# 🦴 Task 1: Bone Segmentation Using Image Processing only

## Task 1.1 – Bone Segmentation (Femur: Red, Tibia: Blue)

1. **Data Loading**

   * Load `.nii.gz` CT scans using `nibabel`.

2. **Preprocessing**

   * Normalize CT intensities to \[0, 1].

3. **Initial Segmentation**

   * Apply Otsu’s method for bone-soft tissue thresholding.

4. **Morphological Processing**

   * Remove noise, apply dilation and closing to clean masks.

5. **Watershed Preparation**

   * Compute distance transform to find bone centers.

6. **Region Segmentation**

   * Use watershed with peak markers to separate bones.

7. **Bone Classification**

   * Identify femur and tibia based on volume and z-coordinate:

     * Higher z → Femur
     * Lower z → Tibia

8. **Output Generation**

   * Save individual bone masks (`.nii.gz`)
   * Create:

     * 2D CT overlays
     * 3D scatter plots

---

## Task 1.2 – Contour Expansion

1. **Data Loading**

   * Load CT and mask `.nii.gz` files, extract voxel spacing.

2. **Structuring Element Creation**

   * Convert mm to voxel units.
   * Generate ellipsoidal kernel accounting for anisotropy.

3. **Mask Expansion**

   * Apply binary dilation using ellipsoidal kernel.
   * Convert to `uint8`.

4. **Output Generation**

   * Save mask with `_expanded_Xmm` suffix.
   * Preserve original metadata.

5. **Visualization**

   * Resample for accurate overlay.
   * Visualize CT with expanded mask.
   * Use femur/tibia-specific filenames.

---

## Task 1.3 – Randomized Contour Adjustment

1. **Initial Expansion**

   * Create full expansion mask using ellipsoidal structuring element.

2. **Random Selection**

   * Identify new voxels from expansion.
   * Randomly select a fraction (`random_fraction`).
   * Combine with original mask.

3. **Output**

   * Save mask with expansion and fraction info.
   * Generate overlay visualizations.

**Parameters**:

* `max_expansion_mm`: Max expansion distance (mm)
* `random_fraction`: 0.0–1.0 proportion of expansion to keep

---

## Task 1.4 – Landmark Detection on Tibia

1. **Coordinate Extraction**

   * Extract voxel coordinates and convert to world coordinates.

2. **Landmark Detection**

   * Sort by Z (inferior → superior).
   * Split by median X (medial vs lateral).
   * Find lowest point in each region.

3. **Multi-Mask Analysis**

   * Compare across:

     * Original
     * 2mm and 4mm expansions
     * Random variants

4. **Output Generation**

   * Print coordinates to console.
   * Export landmark data to CSV.

