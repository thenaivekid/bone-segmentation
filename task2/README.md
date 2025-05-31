# Binary classification of high dimensional tabular data

1. `preprocess_feature_engg.py` contains code for preprocessing and feature extraction
2. `eda.py` includes the code i used for my eda
3. `train_eval.py` includes code for training, grid searching different algorithms and hyperparameters
4. `do_inference.py` contains code used for final  inference and saving results to csv file
5. `view_experiment_logs.py` contains code i used to view the results of training.
# Sub-task 1: Predicted Probabilities

The predicted probabilities for each algorithm and dataset are stored in the `./TASK_2/final_predictions` directory.

### Logistic Regression
- `logistic_regression_train_predictions.csv`
- `logistic_regression_test_predictions.csv`
- `logistic_regression_blind_test_predictions.csv`

### Random Forest
- `random_forest_train_predictions.csv`
- `random_forest_test_predictions.csv`
- `random_forest_blind_test_predictions.csv`

### Support Vector Classifier (SVC)
- `svc_train_predictions.csv`
- `svc_test_predictions.csv`
- `svc_blind_test_predictions.csv`

# Sub-task 2: Documentation of Methodology
## Data preprocessing & feature engineering steps
### EDA
First, I loaded the train and test sets using pandas and played around to familiarize myself with the task at hand.
- I could not make anything out of it about the what kind of data the table represented in real world
- I saw only 315 examples with extremely high dimentions(many cols)
- There is slight class imbalance in both train and test set
- There are ~37% of the values missing in Feature_1712 to Feature_1734
- High-cardinality features (> 150 unique values): 2945
- Constant features (only one unique value): 111
- Useful features (≤ 150 unique values): 184

### Preprocessing and Feature Engineering
- Load the csv files in dataframe using pandas library
- Drop the columns that has contant values in all rows
- Drop the columns with unique values greater than `high_cardinality_threshold`
- Since some columns have 37% missing values and we have so many cols, drop those columns with missing values
- Drop columns that are too correlated to other column i.e. `corr > corr_threshold`
- Optionally, use random forest to find the importances of the features and drop the less important features
- Use PCA to reduce dimentionality of preprocessed data
- Standard scaler is used to bring the values in all columns to same scale



## Model architectures / key hyper-parameters

###### Logistic Regression
- **C**: Regularization strength (lower = more regularization)
- **penalty**: Regularization type ('l1', 'l2', 'elasticnet', 'none')
- **solver**: Optimization algorithm ('liblinear', 'lbfgs', 'saga', 'sag', 'newton-cg')
- **max_iter**: Maximum iterations for convergence

###### Random Forest
- **n_estimators**: Number of trees in the forest
- **max_depth**: Maximum depth of trees (None for unlimited)
- **min_samples_split**: Minimum samples required to split internal node
- **min_samples_leaf**: Minimum samples required at leaf node
- **max_features**: Number of features considered for best split ('sqrt', 'log2', int, float)
- **bootstrap**: Whether to use bootstrap sampling
- **random_state**: Seed for reproducibility

###### Support Vector Classifier (SVC)
- **C**: Regularization parameter (higher = less regularization)
- **kernel**: Kernel function ('linear', 'poly', 'rbf', 'sigmoid')
- **gamma**: Kernel coefficient for 'rbf', 'poly', 'sigmoid' ('scale', 'auto', float)
- **degree**: Polynomial degree (only for 'poly' kernel)
- **probability**: Enable probability estimates (needed for predict_proba)
- **class_weight**: Handle imbalanced classes ('balanced', dict, None)


## Cross-validation scheme
- Since the size of dataset is small, I used 10 fold cross validation to keep as much examples I can in the training set.
- cross validation is done along with grid search for hyperparameter tuning
- Then I use the given test set to further see the generalization of the model

## Results table with the metrics above


#### Logistic Regression

##### Train Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.6444 |
| AUROC | 0.7111 |
| Recall | 0.7016 |
| Specificity | 0.6073 |
| F1-Score | 0.6084 |

##### Validation Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.6800 |
| AUROC | 0.6819 |
| Recall | 0.7381 |
| Specificity | 0.6379 |
| F1-Score | 0.6596 |

#### Random Forest

##### Train Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.8825 |
| AUROC | 0.9630 |
| Recall | 0.8952 |
| Specificity | 0.8743 |
| F1-Score | 0.8571 |

##### Validation Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.6700 |
| AUROC | 0.7053 |
| Recall | 0.5952 |
| Specificity | 0.7241 |
| F1-Score | 0.6024 |

#### Support Vector Classifier (SVC)

##### Train Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.5937 |
| AUROC | 0.6953 |
| Recall | 0.8226 |
| Specificity | 0.4450 |
| F1-Score | 0.6145 |

##### Validation Metrics 
| Metric | Value |
|--------|-------|
| Accuracy | 0.6400 |
| AUROC | 0.6769 |
| Recall | 0.8571 |
| Specificity | 0.4828 |
| F1-Score | 0.6667 |


#### Ensemble Model(Equal voting of top 3 algorithms' models) Results

##### Train Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.6635 |
| AUROC | 0.6635 |
| Recall | 0.8226 |
| Specificity | 0.5602 |
| F1-Score | 0.6581 |

##### Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy | 0.6800 |
| AUROC | 0.6800 |
| Recall | 0.7857 |
| Specificity | 0.6034 |
| F1-Score | 0.6735 |

## A short discussion of strengths, limitations, and how you would improve the model with more time.

### Strengths
- Proper preprocessing
- Uses light weight traditional machine learning methods only
- Uses grid search with 10 fold cross validation for hyper parameter tuning

### Weakness
- The domain knowledge is not used for more powerful feature engineering
- Models are still over fitting to training data
- Overall f1 scores are not satisfactory to deploy in real world scenario
### Future Improvements
- Use clustering to group the features and do some feature engineering to achieve higher f1 scores.
- Understand the domain of the data and try to use the semantic meaning of columns for feature engineering
- Use ensemble of different types of algorithms to vote for better f1 scores














<!-- ## TODO

- [x] handle unbalanced dataset
- [x] grid search for hyperparameter tuning for best 3 algos
- [x] do inference on 3 best models with higher tolerance for high cardinality like 200 +
- [x] write to do inference on the best 3 models
    - [x] save table that stores key hyper parameters 5 eval metrics for each algos
- [x] write code to save the result on blind test set
- [x] write code to use the ensemble of different types of model. use 3 top performing models to vote and get the average and see if it improves the test f1 score
- [ ] save models -->
