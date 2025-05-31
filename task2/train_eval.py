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
    
    # print(f"{X_train.shape=}, {X_val.shape=}")
    
    
    print("Starting Grid Search...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    # print(f"\nBest Parameters: {grid_search.best_params_}")
    # print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
    
    train_metrics = evaluate_model(best_model, X_train, y_train)
    val_metrics = evaluate_model(best_model, X_val, y_val)
    
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

    # print("\n", "*" * 50)  
    # print(f"Overfitting Analysis:")
    # print(f"Training F1: {train_metrics['f1']:.4f}")
    # print(f"Validation F1: {val_metrics['f1']:.4f}")
    # print(f"Train-Val Gap: {train_val_gap:.4f}")  
    # print("*" * 50)  
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
        # {
        #     "name": "PCA_40",
        #     "high_cardinality_threshold": 100,
        #     "corr_threshold": 0.9,
        #     "use_random_forest_selector": False,
        #     "n_components": 40
        # },
        # {
        #     "name": "PCA_25",
        #     "high_cardinality_threshold": 100,
        #     "corr_threshold": 0.9,
        #     "use_random_forest_selector": False,
        #     "n_components": 25
        # },
        # {
        #     "name": "PCA_20",
        #     "high_cardinality_threshold": 98,
        #     "corr_threshold": 0.9,
        #     "use_random_forest_selector": False,
        #     "n_components": 20
        # },
        # {
        #     "name": "PCA_15",
        #     "high_cardinality_threshold": 98,
        #     "corr_threshold": 0.9,
        #     "use_random_forest_selector": False,
        #     "n_components": 15
        # },
        {
            "name": "PCA_10",
            "high_cardinality_threshold": 99,
            "corr_threshold": 0.9,
            "use_random_forest_selector": False,
            "n_components": 7# tested [5, 7, 10, 15, 20, 25, 30, 40]
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
    ## the best #  PCA_10 val f1:0.673684 {'C': 0.1, 'class_weight': 'balanced', 'l1_ratio': 0.9, 'max_iter': 1000, 'penalty': 'elasticnet', 'solver': 'saga'}
    param_grid = {
        # 'C': [0.05, 0.1, 0.5, ],  
        'C': [0.1, ],  
        # 'penalty': ['l1', 'l2', 'elasticnet'],
        'penalty': [ 'elasticnet'], 
        'solver': ['liblinear', 'saga'], 
        'solver': ['saga'], 
        'class_weight': ['balanced'],
        'max_iter': [1000,],
        # 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9], 
        'l1_ratio': [0.8, 0.9], 

        # 'tol': [1e-4, 1e-6] 
    }
    

    base_model = LogisticRegression(random_state=42)
    results_df = run_multiple_configurations(base_model, param_grid, save_file="TASK_2/logistic_regression_results.csv")
    return results_df


def train_svc():
    """Train a Support Vector Classifier with default parameters"""
    # best_config['config_name']='PCA_20'
    # best_config['val_f1']=0.6667
    # best_config['best_params']={'C': 0.1, 'class_weight': 'balanced', 'coef0': 0.1, 'degree': 2, 'gamma': 'scale', 'kernel': 'sigmoid'}
    # (best_config['train_f1'] - best_config['val_f1'])=-0.0522
    param_grid = {
        # 'C': [0.05, 0.5, 0.1,],  
        'C': [0.5, 0.1,],  
        # 'kernel': ['linear', 'rbf', 'poly', "sigmoid"],  
        'kernel': ['poly', "sigmoid"],  
        # 'kernel': ['linear'],  
        # 'gamma': ['scale', 'auto'],  
        'gamma': ['scale', 0.001, 0.01, 0.1,],  
        # 'class_weight': ['balanced', None],
        'class_weight': ['balanced'],
        # 'degree': [2, 3, 4],
        'degree': [2, ],
        'coef0': [0.0, 0.1, 1.0],
    }
    
    from sklearn.svm import SVC
    base_model = SVC(probability=True, random_state=42)
    
    results_df = run_multiple_configurations(base_model, param_grid, save_file="TASK_2/svc_results.csv")
    return results_df
    

def train_decision_tree():

    """Train a Decision Tree Classifier with default parameters"""
    #     BEST CONFIG:
    # best_config['config_name']='Aggressive_Filtering'
    # best_config['val_f1']=0.5800
    # best_config['best_params']={'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 5, 'min_samples_split': 2}
    # (best_config['train_f1'] - best_config['val_f1'])=0.0759
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'class_weight': ['balanced', None],
    }
    
    from sklearn.tree import DecisionTreeClassifier
    base_model = DecisionTreeClassifier(random_state=42)
    
    results_df = run_multiple_configurations(base_model, param_grid, save_file="TASK_2/decision_tree_results.csv")
    return results_df

def train_random_forest():
    """Train a Random Forest Classifier with default parameters"""
    
    # best_config['config_name']='RF_Selector'
    # best_config['val_f1']=0.6024
    # best_config['best_params']={'bootstrap': False, 'class_weight': 'balanced', 'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 50}
    # (best_config['train_f1'] - best_config['val_f1'])=0.2547
    param_grid = {
        # 'n_estimators': [50, 100, 200,300],
        'n_estimators': [5, 10, 20, 50],
        # 'max_depth': [5, 10, 15, 20],
        'max_depth': [1, 2, 3, 5,],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5],
        # 'min_samples_leaf': [1, 2],
        'min_samples_leaf': [2],
        'bootstrap': [ False],
        'class_weight': ['balanced'],
    }
    
    from sklearn.ensemble import RandomForestClassifier
    base_model = RandomForestClassifier(random_state=42)
    
    results_df = run_multiple_configurations(base_model, param_grid, save_file="TASK_2/random_forest_results.csv")
    return results_df



def train_xgboost():
    """Train an XGBoost Classifier with default parameters"""
    import xgboost as xgb
    param_grid = {
        'n_estimators': [300],
        # 'n_estimators': [50,20, 10,],
        'max_depth': [3],
        # 'learning_rate': [3, 0.2, 1],
        # 'subsample': [0.8, 1.0],
        # 'colsample_bytree': [0.8, 1.0],
        # 'gamma': [0, 0.1, 0.2],
        # 'reg_alpha': [0.1, 1,5],
        # 'reg_lambda': [1, 5, 10],
        # 'scale_pos_weight': [1],
        # 'min_child_weight': [5],
    }
    base_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    results_df = run_multiple_configurations(base_model, param_grid, save_file="TASK_2/xgboost_results.csv")
    return results_df


def save_predictions_to_csv(X, model, csv_filename, id_column=None):
   """
   Generate predictions and save class probabilities to CSV
   
   Parameters:
   -----------
   X : array-like or DataFrame
       Processed feature data for inference
   model : trained sklearn model
       Already trained model with predict_proba method
   csv_filename : str
       Name of the CSV file to save predictions
   id_column : array-like, optional
       ID values for each sample. If None, will create sequential IDs
   
   Returns:
   --------
   pandas.DataFrame : DataFrame with predictions saved to CSV
   """
   import pandas as pd
   import numpy as np
   
   # Get class probabilities
   try:
       probabilities = model.predict_proba(X)
   except AttributeError:
       raise ValueError("Model must have predict_proba method for probability predictions")
   
   # Get class labels
   if hasattr(model, 'classes_'):
       class_labels = model.classes_
   else:
       # Fallback for models without classes_ attribute
       class_labels = [f'class_{i}' for i in range(probabilities.shape[1])]
   
   # Create ID column if not provided
   if id_column is None:
       id_column = np.arange(len(X))
   
   # Create DataFrame
   results_df = pd.DataFrame()
   results_df['ID'] = id_column
   
   # Add probability columns for each class
   for i, class_label in enumerate(class_labels):
       results_df[f'prob_class_{class_label}'] = probabilities[:, i]
   
   # Add predicted class column (optional, but often useful)
   predicted_classes = model.predict(X)
   results_df['predicted_class'] = predicted_classes
   
   # Save to CSV
   results_df.to_csv(csv_filename, index=False)
   
   print(f"Predictions saved to {csv_filename}")
   print(f"Shape: {results_df.shape}")
   print(f"Columns: {list(results_df.columns)}")
   
   return results_df

# Example usage function for your workflow
def generate_all_predictions(X_test, trained_models_dict, output_dir="TASK_2/predictions"):
   """
   Generate predictions for multiple trained models
   
   Parameters:
   -----------
   X_test : array-like
       Test/inference data
   trained_models_dict : dict
       Dictionary with model_name: trained_model pairs
   output_dir : str
       Directory to save prediction files
   """
   import os
   
   # Create output directory if it doesn't exist
   os.makedirs(output_dir, exist_ok=True)
   
   all_predictions = {}
   
   for model_name, model in trained_models_dict.items():
       csv_filename = f"{output_dir}/{model_name}_predictions.csv"
       
       try:
           predictions_df = save_predictions_to_csv(
               X=X_test,
               model=model,
               csv_filename=csv_filename
           )
           all_predictions[model_name] = predictions_df
           
       except Exception as e:
           print(f"Error generating predictions for {model_name}: {str(e)}")
   
   return all_predictions


if __name__ == "__main__":
    results_df = train_logistic_regression()
    results_df = train_svc()
    results_df = train_random_forest()
    # results_df = train_decision_tree()
    # results_df = train_xgboost()
   
    best_config = results_df.iloc[0]
    print(f"\nBEST CONFIG:")
    print(f"{best_config['config_name']=}")
    print(f"{best_config['val_f1']=:.4f}")
    print(f"{best_config['best_params']=}")
    print(f"{(best_config['train_f1'] - best_config['val_f1'])=:.4f}")