from sklearn.metrics import f1_score
from preprocess_feature_engg import preprocess_feature_engg
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import os

def train_and_predict_all_models(train_csv, test_csv, blind_test_csv, output_dir="predictions"):
    """
    Train three best-performing models and generate predictions for all datasets
    
    Parameters:
    -----------
    train_csv : str
        Path to training dataset CSV
    test_csv : str
        Path to test dataset CSV  
    blind_test_csv : str
        Path to blind test dataset CSV
    output_dir : str
        Directory to save prediction CSV files
        
    Returns:
    --------
    dict : Dictionary containing trained models and prediction results
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_configs = {
        'logistic_regression': {
            'preprocessing': {
                'high_cardinality_threshold': 99,
                'corr_threshold': 0.9,
                'use_random_forest_selector': False,
                'n_components': 7 
            },
            'model': LogisticRegression(
                C=0.1,
                class_weight='balanced',
                l1_ratio=0.9,
                max_iter=1000,
                penalty='elasticnet',
                solver='saga',
                random_state=42
            )
        },
        'svc': {
            'preprocessing': {
                'high_cardinality_threshold': 98,
                'corr_threshold': 0.9,
                'use_random_forest_selector': False,
                'n_components': 20 
            },
            'model': SVC(
                C=0.1,
                class_weight='balanced',
                coef0=0.1,
                degree=2,
                gamma='scale',
                kernel='sigmoid',
                probability=True,
                random_state=42
            )
        },
        'random_forest': {
            'preprocessing': {
                'high_cardinality_threshold': 150,
                'corr_threshold': 0.9,
                'use_random_forest_selector': True,
                'n_components': None 
            },
            'model': RandomForestClassifier(
                bootstrap=False,
                class_weight='balanced',
                max_depth=5,
                min_samples_leaf=2,
                min_samples_split=5,
                n_estimators=50,
                random_state=42
            )
        }
    }
    
    results = {}
    trained_models = {}
    # Store last y_train and y_test for evaluation
    last_y_train, last_y_test = None, None

    for model_name, config in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            print(len(preprocess_feature_engg(
                train_csv=train_csv,
                val_csv=test_csv,
                test_csv=blind_test_csv,
                **config['preprocessing']
            )))
            X_train, y_train, X_test, y_test, X_blinded_test = preprocess_feature_engg(
                train_csv=train_csv,
                val_csv=test_csv,
                test_csv=blind_test_csv,
                **config['preprocessing']
            )
            
            print(f"Training data shape: {X_train.shape}")
            print(f"Test data shape: {X_test.shape}")
            
            model = config['model']
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            
            datasets = {
                'train': (X_train, f"{model_name}_train_predictions.csv", y_train),
                'test': (X_test, f"{model_name}_test_predictions.csv", y_test),
                'blind_test': (X_blinded_test, f"{model_name}_blind_test_predictions.csv", None),
            }
            
            
            model_results = {}
            
            for dataset_name, (X_data, filename, y_true) in datasets.items():
                print(f"Generating predictions for {dataset_name} set...")
                
                probabilities = model.predict_proba(X_data)
                predictions = model.predict(X_data)
                
                results_df = pd.DataFrame()
                results_df['ID'] = range(len(X_data))
                
                for i, class_label in enumerate(model.classes_):
                    results_df[f'prob_class_{class_label}'] = probabilities[:, i]
                
                results_df['predicted_class'] = predictions
                
                filepath = os.path.join(output_dir, filename)
                results_df.to_csv(filepath, index=False)
                
                model_results[dataset_name] = {
                    'predictions_df': results_df,
                    'filepath': filepath
                }
                
                print(f"Saved predictions to: {filepath}")
                print(f"Shape: {results_df.shape}")
                if y_true is not None:
                    print("*"*50)
                    print(f"{model_name} {dataset_name} f1 score: {f1_score(y_true, predictions):.4f}")
            results[model_name] = model_results
            print(f"{model_name} completed successfully")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
            continue


    return {
        'trained_models': trained_models,
        'predictions': results,
        'output_directory': output_dir
    }






if __name__ == "__main__":
    results = train_and_predict_all_models(
        train_csv="TASK_2/train_set.csv",
        test_csv="TASK_2/test_set.csv", 
        blind_test_csv="TASK_2/blinded_test_set.csv",
        output_dir="TASK_2/final_predictions"
    )
    
    print(f"\nsuccess")
