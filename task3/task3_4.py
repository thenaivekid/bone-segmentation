import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import os
import glob

class FeatureComparator:
    def __init__(self, features_dir="features"):
        self.features_dir = Path(features_dir)
        self.regions = ['tibia', 'femur', 'background']
        self.pairs = [
            ('tibia', 'femur'),
            ('tibia', 'background'),
            ('femur', 'background')
        ]
        
    def load_features(self):       
        feature_files = list(self.features_dir.glob("*.npy"))
                
        print(f"Found {len(feature_files)} feature files:")
        
        layer_groups = {}
        for file_path in feature_files:
            filename = file_path.stem
            parts = filename.split('_')
            
            region = parts[0]
            layer_info = '_'.join(parts[1:-1])
            
            if layer_info not in layer_groups:
                layer_groups[layer_info] = {}
            
            feature_vector = np.load(file_path)
            layer_groups[layer_info][region] = feature_vector
            
            print(f"  {filename}: {feature_vector.shape}")
        print(f"{layer_groups=}")
        return layer_groups
    
    def compute_cosine_similarities(self, features_by_layer):
        results = []
        
        print("\nComputing cosine similarities...")
        
        layer_names = sorted(features_by_layer.keys())
        
        for region1, region2 in self.pairs:
            pair_name = f"{region1}_vs_{region2}"
            row_data = {'comparison': pair_name}
            
            for layer_name in layer_names:
                layer_features = features_by_layer[layer_name]
                
                if region1 in layer_features and region2 in layer_features:
                    feat1 = layer_features[region1].reshape(1, -1)
                    feat2 = layer_features[region2].reshape(1, -1)
                    
                    similarity = cosine_similarity(feat1, feat2)[0, 0]
                    row_data[layer_name] = similarity
                    
                    print(f"{pair_name} - {layer_name}: {similarity:.4f}")
                else:
                    print(f"Missing features for pair {region1}-{region2} in layer {layer_name}")
                    row_data[layer_name] = "N/A"
            
            results.append(row_data)
        
        return results
    
    def save_results_to_csv(self, results, output_file="cosine_similarities.csv"):      
        df = pd.DataFrame(results)
        
        output_path = Path(output_file)
        df.to_csv(output_path, index=False, float_format='%.6f')
        
        print(f"\nResults saved to: {output_path}")

def main():
    print("Starting Feature Comparison Analysis...")
    print("="*50)
    
    comparator = FeatureComparator()
    
    try:
        print("Loading features...")
        features_by_layer = comparator.load_features()
        
        if not features_by_layer:
            print("No features loaded!")
            return
        
        print(f"\nLoaded features from {len(features_by_layer)} layers")
        
        results = comparator.compute_cosine_similarities(features_by_layer)
        
        if not results:
            print("No similarities computed!")
            return
        
        df = comparator.save_results_to_csv(results)
        
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()