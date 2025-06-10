import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from task3_1 import load_ct_scan
from task3_2 import get_loaded_3d_model

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.features = {}
        self.hooks = []
        self.conv_layers = self._get_conv_layers()
        self._register_hooks()
        
    def _get_conv_layers(self):
        """Get all convolution layers in the model"""
        conv_layers = []
        
        def find_conv_layers(module, prefix=''):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, nn.Conv3d):
                    conv_layers.append((full_name, child))
                else:
                    find_conv_layers(child, full_name)
        
        find_conv_layers(self.model)
        print(f"Found {len(conv_layers)} Conv3d layers")
        for i, (name, _) in enumerate(conv_layers[-10:]): 
            print(f"  Layer {len(conv_layers)-10+i}: {name}")
        
        return conv_layers
    
    def _register_hooks(self):
        """Register hooks for last, 3rd-last, and 5th-last conv layers"""
        target_indices = [-1, -3, -5]
        target_layers = [self.conv_layers[i] for i in target_indices]
        
        print("\nRegistering hooks for:")
        for idx, (name, layer) in zip(target_indices, target_layers):
            print(f"  {idx}: {name}")
            hook = layer.register_forward_hook(self._make_hook(name))
            self.hooks.append(hook)
    
    def _make_hook(self, layer_name):
        """Create a hook function for a specific layer"""
        def hook(module, input, output):
            self.features[layer_name] = output.detach()
        return hook
    
    def extract_features(self, x):
        """Extract features from input tensor"""
        self.features.clear()
        
        with torch.no_grad():
            _ = self.model.features(x)
        
        gap_features = {}# global avg pooling
        for layer_name, feature_map in self.features.items():
            gap = torch.mean(feature_map, dim=(2, 3, 4)) 
            gap_features[layer_name] = gap.cpu().numpy()
            print(f"{layer_name}: {feature_map.shape} -> GAP: {gap.shape}")
        
        return gap_features
    
    def cleanup(self):
        """Remove hooks"""
        for hook in self.hooks:
            hook.remove()

def process_segmentation_data(data):
    """Process segmentation data for model input"""
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
    tensor = tensor.permute(0,1,4,2,3)
    return tensor

def save_features(features_dict, region_name, save_dir):
    """Save extracted features to files"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    for layer_name, features in features_dict.items():
        clean_name = layer_name.replace('.', '_')
        filename = f"{region_name}_{clean_name}_features.npy"
        filepath = save_dir / filename
        
        np.save(filepath, features)
        print(f"Saved {region_name} features from {layer_name}: {features.shape} -> {filepath}")

def main():
    model = get_loaded_3d_model()
    
    extractor = FeatureExtractor(model)
    

    tibia_data, _ = load_ct_scan("/teamspace/studios/this_studio/bone-segmentation/task3/results/tibia_segmentation.nii.gz")
    femur_data, _ = load_ct_scan("/teamspace/studios/this_studio/bone-segmentation/task3/results/femur_segmentation.nii.gz")
    background_data, _ = load_ct_scan("/teamspace/studios/this_studio/bone-segmentation/task3/results/background_segmentation.nii.gz")
    
    features_dir = "features"
    os.makedirs(features_dir, exist_ok=True)
    
    regions = {
        'tibia': tibia_data,
        'femur': femur_data,
        'background': background_data
    }
    
    for region_name, data in regions.items():
        print(f"\n{'='*50}")
        print(f"Processing {region_name} ")
        
        try:
            input_tensor = process_segmentation_data(data)
            print(f"Model input tensor shape: {input_tensor.shape}")

            features = extractor.extract_features(input_tensor)

            save_features(features, region_name, features_dir)
            
        except Exception as e:
            print(f"Error processing {region_name}: {str(e)}")
            continue
    
    extractor.cleanup()
    print(f"\nFeature extraction completed!")

if __name__ == "__main__":
    main()