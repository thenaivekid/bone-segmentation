import pandas as pd

files = [
    ("/workspaces/bone-segmentation/task2/TASK_2/logistic_regression_results.csv", "Logistic Regression"),
    ("/workspaces/bone-segmentation/task2/TASK_2/random_forest_results.csv", "Random Forest"),
    ("/workspaces/bone-segmentation/task2/TASK_2/svc_results.csv", "SVC")
]

train_cols = ['train_accuracy', 'train_auroc', 'train_recall', 'train_specificity', 'train_f1']
val_cols = ['val_accuracy', 'val_auroc', 'val_recall', 'val_specificity', 'val_f1']

for file_path, model_name in files:
    df = pd.read_csv(file_path)
    print(f"{model_name} - TRAIN METRICS:")
    print("-" * 50)
    print(df[train_cols].round(4).to_string(index=False))
    print(f"{model_name} - VALIDATION METRICS:")
    print("-" * 50)
    print(df[val_cols].round(4).to_string(index=False))
    print()