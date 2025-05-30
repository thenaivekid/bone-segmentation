import pandas as pd

df = pd.read_csv("/teamspace/studios/this_studio/task2/results/baselines.csv")
print(df[["model", "val_accuracy", "train_accuracy", "val_auroc", "train_auroc", "val_recall", "train_recall", "val_specificity", "train_specificity", "val_f1", "train_f1"]])