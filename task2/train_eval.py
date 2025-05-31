from preprocess_feature_engg import preprocess_feature_engg
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn + fp)

    return {
        "accuracy": acc,
        "auroc": auc,
        "recall": recall,
        "specificity": specificity,
        "f1": f1
    }

def train_model_with_grid_search(base_model, param_grid, high_cardinality_threshold=100, corr_threshold=1, 
                                use_random_forest_selector=False, n_components=None):
    
    X_train, y_train, X_val, y_val = preprocess_feature_engg(
        train_csv="TASK_2/train_set.csv",
        val_csv="TASK_2/test_set.csv",
        high_cardinality_threshold=high_cardinality_threshold,
        corr_threshold=corr_threshold,
        use_random_forest_selector=use_random_forest_selector,
        n_components=n_components
    )
    
    print(f"{X_train.shape=}, {X_val.shape=}")
    
    
    print("Starting Grid Search...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
    
    # Evaluate on train and validation sets
    train_metrics = evaluate_model(best_model, X_train, y_train)
    val_metrics = evaluate_model(best_model, X_val, y_val)
    
    # Create detailed results
    result = {
        "model": "LogisticRegression_GridSearch",
        "best_params": grid_search.best_params_,
        "cv_f1_score": grid_search.best_score_,
        "high_cardinality_threshold": high_cardinality_threshold,
        "corr_threshold": corr_threshold,
        "use_rf_selector": use_random_forest_selector,
        "n_components": n_components,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()}
    }
    
  
    train_val_gap = train_metrics['f1'] - val_metrics['f1']
    print(f"\nOverfitting Analysis:")
    print(f"Training F1: {train_metrics['f1']:.4f}")
    print(f"Validation F1: {val_metrics['f1']:.4f}")
    print(f"Train-Val Gap: {train_val_gap:.4f}")
    
    if train_val_gap > 0.1:
        print("⚠️  High overfitting detected (gap > 0.1)")
    elif train_val_gap > 0.05:
        print("⚠️  Moderate overfitting detected (gap > 0.05)")
    else:
        print("✅ Good generalization (gap <= 0.05)")
    
    return result, best_model, grid_search

def run_multiple_configurations(base_model, param_grid, save_file):
    """Run grid search with different preprocessing configurations"""
    
    configurations = [
        # {
        #     "name": "Base",
        #     "high_cardinality_threshold": 100,
        #     "corr_threshold": 1,
        #     "use_random_forest_selector": False,
        #     "n_components": None
        # },
        # {
        #     "name": "Aggressive_Filtering",
        #     "high_cardinality_threshold": 50,
        #     "corr_threshold": 0.9,
        #     "use_random_forest_selector": False,
        #     "n_components": None
        # },
        
        # {
        #     "name": "RF_Selector",
        #     "high_cardinality_threshold": 150,
        #     "corr_threshold": 0.9,
        #     "use_random_forest_selector": True,
        #     "n_components": None
        # },
        {
            "name": "PCA_50",
            "high_cardinality_threshold": 100,
            "corr_threshold": 0.9,
            "use_random_forest_selector": False,
            "n_components": 50
        },
        {
            "name": "PCA_25",
            "high_cardinality_threshold": 100,
            "corr_threshold": 0.9,
            "use_random_forest_selector": False,
            "n_components": 25
        },
        {
            "name": "PCA_20",
            "high_cardinality_threshold": 100,
            "corr_threshold": 0.9,
            "use_random_forest_selector": False,
            "n_components": 20
        },
        {
            "name": "PCA_15",
            "high_cardinality_threshold": 100,
            "corr_threshold": 0.9,
            "use_random_forest_selector": False,
            "n_components": 15
        },
        {
            "name": "PCA_10",
            "high_cardinality_threshold": 100,
            "corr_threshold": 0.9,
            "use_random_forest_selector": False,
            "n_components": 10
        },
    ]
    
    
    all_results = []
    for config in configurations:
        print(f"Running Configuration: {config['name']}")
        
        try:
            result, best_model, grid_search = train_model_with_grid_search(
                base_model,
                param_grid,
                high_cardinality_threshold=config["high_cardinality_threshold"],
                corr_threshold=config["corr_threshold"],
                use_random_forest_selector=config["use_random_forest_selector"],
                n_components=config["n_components"]
            )
            result["config_name"] = config["name"]
            all_results.append(result)
            
        except Exception as e:
            print(f"Error in configuration {config['name']}: {str(e)}")
            continue
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="val_f1", ascending=False)
    
    results_df.to_csv(save_file, index=False)
    
    print("FINAL RESULTS - Top Configurations by Validation F1:")
    
    display_cols = ["config_name", "val_f1", "train_f1", "cv_f1_score", 
                   "val_accuracy", "val_auroc", "best_params"]
    print(results_df[display_cols].to_string(index=False))
    
    return results_df


def train_logistic_regression():
    """Train a logistic regression model with default parameters"""
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'class_weight': ['balanced', None],
        'max_iter': [1000, 2000]
    }
    

    base_model = LogisticRegression(random_state=42)
    results_df = run_multiple_configurations(base_model, param_grid, save_file="TASK_2/logistic_regression_results.csv")
    return results_df


    
if __name__ == "__main__":
    results_df = train_logistic_regression()
    
    best_config = results_df.iloc[0]
    print(f"\nBEST CONFIG:")
    print(f"{best_config['config_name']=}")
    print(f"{best_config['val_f1']=:.4f}")
    print(f"{best_config['best_params']=}")
    print(f"{(best_config['train_f1'] - best_config['val_f1'])=:.4f}")