from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, confusion_matrix
from preprocess_feature_engg import preprocess_feature_engg

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def train_ensemble_of_top_3_models():
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
    
    trained_models = {}
    val_predictions = {}
    train_predictions = {}
    
    for model_name, config in model_configs.items():
        print(f"\nTraining {model_name}...")
        
        X_train, y_train, X_val, y_val = preprocess_feature_engg(
            train_csv="TASK_2/train_set.csv",
            val_csv="TASK_2/test_set.csv",
            **config['preprocessing']
        )
        
        model = config['model']
        model.fit(X_train, y_train)
        
        train_pred = model.predict(X_train)
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        val_pred = model.predict(X_val)
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        
        train_metrics = {
            "accuracy": accuracy_score(y_train, train_pred),
            "auroc": roc_auc_score(y_train, train_pred_proba),
            "recall": recall_score(y_train, train_pred),
            "specificity": calculate_specificity(y_train, train_pred),
            "f1": f1_score(y_train, train_pred)
        }
        
        val_metrics = {
            "accuracy": accuracy_score(y_val, val_pred),
            "auroc": roc_auc_score(y_val, val_pred_proba),
            "recall": recall_score(y_val, val_pred),
            "specificity": calculate_specificity(y_val, val_pred),
            "f1": f1_score(y_val, val_pred)
        }
        
        print(f"{model_name} - Train metrics:", train_metrics)
        print(f"{model_name} - Test metrics:", val_metrics)
        
        trained_models[model_name] = model
        val_predictions[model_name] = val_pred
        train_predictions[model_name] = train_pred
    
    print("\n" + "="*50)
    print("ENSEMBLE RESULTS")
    print("="*50)
    
    train_ensemble_pred = []
    val_ensemble_pred = []
    
    for i in range(len(list(train_predictions.values())[0])):
        votes = [train_predictions[model][i] for model in train_predictions.keys()]
        train_ensemble_pred.append(1 if sum(votes) >= 2 else 0)
    
    for i in range(len(list(val_predictions.values())[0])):
        votes = [val_predictions[model][i] for model in val_predictions.keys()]
        val_ensemble_pred.append(1 if sum(votes) >= 2 else 0)
    
    train_ensemble_metrics = {
        "accuracy": accuracy_score(y_train, train_ensemble_pred),
        "auroc": accuracy_score(y_train, train_ensemble_pred),
        "recall": recall_score(y_train, train_ensemble_pred),
        "specificity": calculate_specificity(y_train, train_ensemble_pred),
        "f1": f1_score(y_train, train_ensemble_pred)
    }
    
    val_ensemble_metrics = {
        "accuracy": accuracy_score(y_val, val_ensemble_pred),
        "auroc": accuracy_score(y_val, val_ensemble_pred),
        "recall": recall_score(y_val, val_ensemble_pred),
        "specificity": calculate_specificity(y_val, val_ensemble_pred),
        "f1": f1_score(y_val, val_ensemble_pred)
    }
    
    print("Ensemble - Train metrics:", train_ensemble_metrics)
    print("Ensemble - Test metrics:", val_ensemble_metrics)
    
    return trained_models

train_ensemble_of_top_3_models()