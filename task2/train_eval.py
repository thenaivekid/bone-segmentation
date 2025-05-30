from preprocess_feature_engg import preprocess_feature_engg
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, f1_score, confusion_matrix
import pandas as pd

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

def train_model(model, model_name, high_cardinality_threshold=10, corr_threshold=1, use_random_forest_selector=False, n_components=None):
    X_train, y_train, X_val, y_val = preprocess_feature_engg(
        train_csv="/teamspace/studios/this_studio/task2/TASK_2/train_set.csv",
        val_csv="/teamspace/studios/this_studio/task2/TASK_2/test_set.csv",
        high_cardinality_threshold=high_cardinality_threshold,
        corr_threshold=corr_threshold,
        use_random_forest_selector=use_random_forest_selector,
        n_components=n_components
    )
    model.fit(X_train, y_train)

    train_metrics = evaluate_model(model, X_train, y_train)
    val_metrics = evaluate_model(model, X_val, y_val)

    result = {
        "model": model_name,
        "high_cardinality_threshold": high_cardinality_threshold,
        "corr_threshold": corr_threshold,
        "use_rf_selector": use_random_forest_selector,
        "n_components": n_components,
        **{f"train_{k}": v for k, v in train_metrics.items()},
        **{f"val_{k}": v for k, v in val_metrics.items()}
    }
    # print(result)
    return result

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    import xgboost as xgb

    models = [
        # ("SVC linear", SVC(kernel = "linear", probability=True)),
        # ("SVC poly", SVC(kernel = "poly", probability=True)),
        # ("SVC sigmoid", SVC(kernel = "sigmoid", probability=True)),
        # ("SVC rbf", SVC(kernel = "rbf", probability=True)),
        ("LogisticRegression", LogisticRegression(
                                    penalty='l2',           # or 'l1' if you want feature selection
                                    C=1.0,                  # tune this
                                    class_weight='balanced',
                                    solver='liblinear',     # good for small binary datasets
                                    max_iter=1000,
                                    random_state=42
                                )
                                ),
        # ("DecisionTree", DecisionTreeClassifier()),
        # ("RandomForest", RandomForestClassifier()),
        # ("GradientBoosting", GradientBoostingClassifier()),
        # ("XGBoost", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ]

    experiment_logs = []

    for name, model in models:
        print(f"\nRunning experiment for {name}")
        result = train_model(
            model=model,
            model_name=name,
            high_cardinality_threshold=100,
            # corr_threshold=0.9,
            use_random_forest_selector=False,
            n_components=15
        )
        experiment_logs.append(result)

    results_df = pd.DataFrame(experiment_logs)
    results_df = results_df.sort_values(by="val_f1", ascending=False)
    results_df.to_csv("/teamspace/studios/this_studio/task2/results/baselines_n_comp_15.csv")
    print("\nTop Performing Models by Validation F1:")
    print(results_df[["model", "val_accuracy", "val_auroc", "val_recall", "val_specificity", "val_f1", "train_f1"]])
