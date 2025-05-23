# Task 1
## Details of approaches 


### Task 1.1 – Bone Segmentation
 1. Data Loading
- **Load CT Data**: Reads the .nii.gz file using nibabel to extract 3D CT scan data and spatial information

 2. Preprocessing
- **Preprocessing**: Normalizes CT intensity values to 0-1 range for consistent processing

 3. Initial Segmentation
- **Initial Thresholding**: Uses Otsu's method to automatically determine threshold for separating bone from soft tissue

 4. Morphological Processing
- **Morphological Cleanup**: Removes small noise objects and applies closing/dilation operations to fill gaps and smooth bone regions

 5. Watershed Preparation
- **Distance Transform**: Calculates distance from bone boundaries to find bone centers for watershed seeds

 6. Region Segmentation
- **Watershed Segmentation**: Uses distance transform peaks as markers to separate individual bone structures using watershed algorithm

 7. Bone Classification
- **Bone Identification**: Analyzes segmented regions by volume and position to automatically identify the two largest bones (tibia and femur)
- **Anatomical Classification**: Assigns tibia/femur labels based on spatial position (higher z-coordinate = femur, lower = tibia)

 8. Output Generation
- **Output Generation**: 
  - Saves individual bone masks as .nii.gz files
  - Creates 2D slice visualizations showing original CT with colored overlays
  - Generates 3D scatter plot visualization of both bones

### Task 1.2 – Contour Expansion
 1. Data Loading
- **Load CT and Mask Data**: Reads .nii.gz files using nibabel to extract both CT scan and segmentation mask data
- **Extract Voxel Spacing**: Retrieves voxel dimensions from image headers for accurate spatial calculations

 2. Structuring Element Creation
- **Convert mm to Voxels**: Calculates radius in voxel units for each dimension based on voxel spacing
- **Create Ellipsoidal Kernel**: Generates 3D ellipsoid structuring element that maintains true millimeter distances across anisotropic voxels
- **Account for Anisotropy**: Handles different voxel sizes in x, y, z directions for accurate expansion

 3. Mask Expansion
- **Binary Dilation**: Applies morphological dilation using the custom ellipsoidal structuring element
- **Preserve Data Type**: Converts result back to uint8 format for consistent handling

 4. Output Generation
- **Save Expanded Mask**: Creates new .nii.gz file with "_expanded_Xmm" suffix indicating expansion distance
- **Maintain Metadata**: Preserves original affine transformation and header information

 5. Visualization
- **Resample for Display**: Resamples mask to match CT resolution for accurate overlay visualization
- **Generate Overlays**: Creates slice-by-slice visualizations showing original CT with expanded mask overlay
- **Bone-Specific Naming**: Uses different filenames and slice indices for femur vs tibia visualization



### Task 1.3 – Randomized Contour Adjustment

 1. Initial Expansion
- Creates full expansion mask using ellipsoidal structuring element
- Maintains spatial accuracy with real millimeter distances

 2. Random Selection
- Identifies expansion region (new voxels not in original mask)
- Randomly selects subset based on `random_fraction` parameter (e.g., 50%)
- Combines original mask with randomly selected expansion voxels

 3. Output
- Saves randomized mask with descriptive filename indicating expansion distance and fraction
- Generates visualization overlays for quality control

## Parameters
- **max_expansion_mm**: Maximum expansion distance in millimeters
- **random_fraction**: Proportion of expansion region to include (0.0-1.0)

## Use Cases
- **Data Augmentation**: Creating varied training samples from single masks
- **Uncertainty Modeling**: Simulating segmentation uncertainty boundaries
- **Robustness Testing**: Evaluating algorithm performance with imperfect masks


 Task 1.4 – Landmark Detection on Tibia
# Bone Landmark Detection Approach

## Overview
Detects medial and lateral landmark points at the lowest (most inferior) positions of bone masks across different mask variations.

## Key Steps

 1. Coordinate Extraction
- Extracts all voxel coordinates from binary mask
- Converts voxel coordinates to world coordinates using affine transformation

 2. Landmark Detection
- **Sort by Z-axis**: Orders points from inferior to superior (lowest to highest)
- **Split by X-axis**: Uses median X-coordinate to separate medial vs lateral regions
- **Find Lowest Points**: Identifies first (lowest) point in each medial/lateral region

 3. Multi-Mask Analysis
- Processes multiple mask variations:
  - Original segmentation
  - 2mm expansion
  - 4mm expansion  
  - Random expansion variants
- Compares landmark positions across all versions

 4. Output Generation
- **Console Display**: Prints medial/lateral coordinates for each mask type
- **CSV Export**: Saves all landmark coordinates to structured file for analysis
