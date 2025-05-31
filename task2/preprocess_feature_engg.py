import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_feature_engg(train_csv, val_csv, test_csv = None, high_cardinality_threshold=10, corr_threshold=0.8, use_random_forest_selector=False, n_components=None,):
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)

    constant_cols = [col for col in df_train.columns if df_train[col].nunique() <= 1]
    df_train = df_train.drop(columns=constant_cols)
    df_val = df_val.drop(columns=constant_cols)
    print(f"Dropped {len(constant_cols)} constant features.")

    threshold = high_cardinality_threshold
    high_card_cols = [col for col in df_train.columns if df_train[col].nunique() > threshold]
    df_train = df_train.drop(columns=high_card_cols)
    df_val = df_val.drop(columns=high_card_cols)

    print(f"Dropped {len(high_card_cols)} high-cardinality features.")

    df_train = df_train.dropna(axis=1)
    df_val = df_val[df_train.columns]
    print(f"Dropped columns with missing values. Remaining columns: {df_train.shape[1]}")

    X_train_temp = df_train.drop(columns=["CLASS"])
    corr_matrix = X_train_temp.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = []
    for column in upper_tri.columns:
        for row in upper_tri.index:
            if upper_tri.loc[row, column] > corr_threshold:
                var_row = X_train_temp[row].var()
                var_col = X_train_temp[column].var()
                to_drop.append(row if var_row < var_col else column)
    to_drop = list(set(to_drop))
    df_train = df_train.drop(columns=to_drop)
    df_val = df_val.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} highly correlated features (threshold={corr_threshold}).")

    X_train = df_train.drop(columns=["CLASS"])
    y_train = df_train["CLASS"]
    X_val = df_val.drop(columns=["CLASS"])
    y_val = df_val["CLASS"]
    if test_csv:
        df_test = pd.read_csv(test_csv)
        df_test = df_test.drop(columns=constant_cols + high_card_cols + to_drop)
        X_test = df_test[X_val.columns]
    if use_random_forest_selector:
        print("training random forest")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        importances = pd.Series(clf.feature_importances_, index=X_train.columns)
        selected_features = importances[importances > 0.01].sort_values(ascending=False)
        X_train = X_train[selected_features.index]
        X_val = X_val[selected_features.index]
        if test_csv:
            X_test = X_test[selected_features.index]
        print(f"Selected {X_train.shape[1]} important features.")

    if n_components:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        if test_csv:
            X_test = pca.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    if test_csv:
        X_test = scaler.transform(X_test)

    print(f"{X_train.shape=}, {y_train.shape=}, {X_val.shape=}, {y_val.shape=}")
    if test_csv is not None:
        return X_train, y_train, X_val, y_val, X_test  
    else:
        return X_train, y_train, X_val, y_val


if __name__ == "__main__":
    preprocess_feature_engg("/teamspace/studios/this_studio/task2/TASK_2/train_set.csv",
        "/teamspace/studios/this_studio/task2/TASK_2/test_set.csv",
        use_random_forest_selector=True,

    )

    ##### without random forest selector and pca, num features = 9
    ##### without pca, num features = 8