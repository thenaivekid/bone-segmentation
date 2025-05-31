# TODO

- [x] handle unbalanced dataset
- [x] grid search for hyperparameter tuning for best 3 algos
- [ ] do inference on 3 best models with higher tolerance for high cardinality like 200 +
- [] write to do inference on the best 3 models
    - [ ] save table that stores key hyper parameters 5 eval metrics for each algos
- [ ] write code to save the result on blind test set
<!-- - [ ] save models -->

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



## Cross-validation scheme

## Results table with the metrics above

## A short discussion of strengths, limitations, and how you would improve the model with more time.

### Future Improvements
- Use clustering to group the features and do some feature engineering to achieve higher f1 scores.
- Understand the domain of the data and try to use the semantic meaning of columns for feature engineering
- 














