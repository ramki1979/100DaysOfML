# 30DaysOfML
I am taking up the challenge to Learn Machine Learning at least 3 hours a everyday!!

## Day 1 : August 27 , 2018
 
**Today's Progress** : Completed Python Programming round-up on [kaggle](https://www.kaggle.com/learn/python)

**Summary** : 
    I work on different languages in my day job & python is not one, I worked on it few years back but didn't remember much.

    I concentrated more on List, Dictionary, Comprehensions, Imports.

    Trying to get the concepts rather than syntax!!.

## Day 2 : August 28 , 2018

**Today's Progress** : 
##### Morning :
Started working on Machine Learning Course on [kaggle](https://www.kaggle.com/learn/machine-learning)

        
**Summary** : Learned about Models.
    
Get your data, load your data using pandas, that gives you DataFrame object.
    
```python
    # pandas library/framework for loading & cleaning your data
    import pandas as pa

    #path to your data file, assuming our data is in .csv format
    data_file_path = './path_to/data_file.csv'

    #load/read data from the file
    ml_data = pd.read_csv(data_file_path)

    #To build our model we have to know the columns, we get columns by using columns property
    ml_data.columns

    #drop missing data [they are other ways to handle this, we will learn it later]
    #dropping row/record if any column_data is missing, read it as Not Available (na)
    ml_data.dropna(axis=0)
```

Target & Features, Target is the column that we want to predict and Features are the columns that defines the Target column outcome.

Assume that our data is related to house rental price and our columns are 

*noOfRooms*, *yearBuilt*, *isCarParking*, *isPowerBackup*, *houseType*, *rentalPrice*, *avgRentPeroid*

we want to predict rental price of a house, so our Prediction Target is *rentalPrice*
*rentalPrice* is dependent on *noOfRooms*, *yearBuilt*, *isCarParking*, *isPowerBackup*, *houseType*

```python
    y = ml_data.rentalPrice
    ml_features = ['noOfRooms', 'yearBuilt', 'isCarParking', 'isPowerBackup', 'houseType']
    X = ml_data[ml_features]

    #DataFrame object has describe & head methods which are very important to look into our data.
    X.describe()
    X.head()
```
    
We use scikit-learn library/framework for building our model.

```python
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error

    # Define our model, giving random_state a value ensures same results on each run.
    ml_model = DecisionTreeRegressor(random_state=9)

    # fit model
    ml_model.fit(X, y)

    # predict
    predictions = ml_model.predict(X)

    # print our model MAE 
    print(mean_absolute_error(y, predictions))
```

Above we used the same data for training & predicting, which is useless practically. why?
Prediction is done for new data, for existing data the value are already there.

so what we do, split our data into two parts *training* & *validation*
```python
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split

    # splitting data into training & validation data for both features & target
    ml_train_X, ml_train_y, ml_val_X, ml_val_y = train_test_split(X, y, random_state=9)

    # Define our model
    ml_model = DecisionTreeRegressor(random_state=9)

    # fit model
    ml_model.fit(ml_train_X, ml_train_y)

    # get predicted prices on validation data
    val_predictions = ml_model.predict(ml_val_X)

    # print our model MAE
    ml_mae = mean_absolute_error(val_y, val_predictions)

    print(ml_mae)
```

##### Evening :
    
Continuing the [kaggle](https://www.kaggle.com/learn/machine-learning) course

**Summary** :

overFitting is when a model matches the training data almost perfectly, but very poorly on new data.

underFitting is when a model fails to capture important distinctions & patterns in the data and does poorly on training data (implies will do poorly on new data too).

Finding the sweet spot between the two, give us the needed accuracy on new data, i.e finding the best tree size for the data.

using *max_leaf_nodes* of **DecisionTreeRegressor** we can get the best tree size.

```python
# by invoking this function with different max_leaf_nodes will give us best mae for the model
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=9)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

Once we find the value for *max_leaf_nodes* we will re-write the model
```python
    # Fit the model with best tree size. Fill in argument to make optimal size
    ml_final_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=9)

    # Fit model with all of our data as we now know the best tree size for our data.
    ml_final_model.fit(X, y)
```

Till now we are using **Decision Tree models** which are not good by todays ML standards.
Now we will look at **Random Forests** which are based on *Decision Trees* but will give much better results.

Random Forest uses many trees and makes prediction by averaging the prediction of each tree.
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ml_model = RandomForestRegressor(random_state=9)

ml_model.fit(train_X, train_y)
ml_model_predictions = ml_model.predict(val_X)
ml_model_mae = mean_absolute_error(ml_model_predictions, val_y)
```
**for sample iowa data, results for both DT & RF**:

Validation DT, MAE when not specifying max_leaf_nodes   : **29,653**

Validation DT, MAE for best value of max_leaf_nodes     : **26,763**

Validation RF, MAE                                      : **22,762**

By default *RandomForest* gives better results then *DT*, modifying the parameters doesn't improve the results much.

There are other models that gives better accuracy than *RandomForest* but requires good skill to get the parameters right.

With this I completed Level 1 of the course.

## Day 3 : August 29 , 2018

**Today's Progress** : 
##### Morning :
continuing Machine Learning Course on [kaggle](https://www.kaggle.com/learn/machine-learning)

**Summary** : 

Started Level 2. Learned how to handle missing values.

created a modified model of default iowa_data for Level 1 & submitted to kaggle competition.

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer # handling missing data

# Path of the file to read. We changed the directory structure to simplify submitting to a competition
iowa_file_path = '../input/train.csv'
home_data = pd.read_csv(iowa_file_path)

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# The list of columns is stored in a variable called features
features = ['LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'TotRmsAbvGrd', 'GarageArea', 'EnclosedPorch', 'YrSold']

# Create target object and call it y
y = home_data.SalePrice
# Create features, call it X
X = home_data[features]

# create X_test which comes from test_data but includes only the columns you used for prediction.
X_test = test_data[features]

# copy train & test features for handling missing data
imputed_X_train = X.copy()
imputed_X_test = X_test.copy()

cols_with_missing = (col for col in X.columns 
                                 if X[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train[col + '_was_missing'] = imputed_X_train[col].isnull()
    imputed_X_test[col + '_was_missing'] = imputed_X_test[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(imputed_X_train)
imputed_X_test = my_imputer.transform(imputed_X_test)

# create a new Random Forest model which we train on all training data
rf_model = RandomForestRegressor()

# fit rf_model on all data from the 
rf_model.fit(imputed_X_train, y)

# make predictions which we will submit. 
test_preds = rf_model.predict(imputed_X_test)

# The lines below shows you how to save your data in the format needed to score it in the competition
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
```

I submitted the above model to the kaggle competition and Ranked **56**

##### Evening :
    
Continuing the [kaggle](https://www.kaggle.com/learn/machine-learning) course

**Summary** : 

Handling Missing Value in Dataset can be handle in 3 ways

```python
# drop columns with missing values
cols_with_missing = [col for col in original_data.columns 
                                 if original_data[col].isnull().any()]
redued_original_data = original_data.drop(cols_with_missing, axis=1)
reduced_test_data = test_data.drop(cols_with_missing, axis=1)

# Imputation fills missing values with some number, this is better than dropping the column
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
data_with_imputed_values = my_imputer.fit_transform(original_data)

# Extension to Imputation, I used in the above submitted model. 
# this will help in some cases & fails in other cases
```

Using categorical data with One Hot Encoding.
Till now we only considered number columns for prediction, we don't know how to use string type, now we will handle exactly that.

Categorical data means, the column represent a category ex: car brands, colors etc.

One Hot Encoding creates indvidual (binary) columns for each value/item of the category and indicates it presence in the row with 1 and absence with 0.

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))

# Applying to Multiple Files, like train dataset file & test dataset file
# Ensure the test data is encoded in the same manner as the training data with the align command
one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)
```

The world is filled with categorical data. You will be a much more effective data scientist if you know how to use this data. Here are resources that will be useful as you start doing more sophisticated work with cateogircal data.

**Pipelines**: Deploying models into production ready systems is a topic unto itself. While one-hot encoding is still a great approach, your code will need to built in an especially robust way. Scikit-learn pipelines are a great tool for this. Scikit-learn offers a class for one-hot encoding and this can be added to a Pipeline. Unfortunately, it doesn't handle text or object values, which is a common use case.

**Applications To Text for Deep Learning**: Keras and TensorFlow have fuctionality for one-hot encoding, which is useful for working with text.

**Categoricals with Many Values**: Scikit-learn's FeatureHasher uses the hashing trick to store high-dimensional data. This will add some complexity to your modeling code.

**XGBoost**:
 
XGBoost is the leading model for working with standard tabular data.
XGBoost is a *Gradient Boosted Decision Tree* that looks like below:

![XGBoost algorithm diagram](https://i.imgur.com/e7MIgXk.png)

```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

data = pd.read_csv('../input/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

from xgboost import XGBRegressor

my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)

# make predictions
predictions = my_model.predict(test_X)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))

# using n_estimators & early_stopping_rounds
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

# using n_estimators, learning_rate & early_stopping_rounds
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)

```
*XGBoost* has few parameters that can greately effect the model accuracy & training speed.

*n_estimators*: specifies number of times to go through the circle describe above
*early_stopping_rounds*: it causes the iterator/cycling to stop when the validation score stops improving.

It's is smart to give large *n_estimators* value and then use *early_stopping_rounds* to find the optimal model accuracy.

5 is a reasonable value for *early_stopping_rounds*

When using *early_stopping_rounds*, you need to set aside some of your data for checking the number of rounds to use. If you later want to fit a model with all of your data, set *n_estimators* to whatever value you found to be optimal when run with early stopping.

*learning_rate*
Instead of getting predictions by simply adding up the predictions from each component model, we will multiply the predictions from each model by a small number before adding them in. This means each tree we add to the ensemble helps us less. In practice, this reduces the model's propensity to overfit.

So, you can use a higher value of *n_estimators* without overfitting. If you use early stopping, the appropriate number of trees will be set automatically.

In general, a small *learning_rate* (and large number of estimators) will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations through the cycle.

*n_jobs*
On larger datasets where runtime is a consideration, you can use parallelism to build your models faster. It's common to set the parameter *n_jobs* equal to the number of cores on your machine. On smaller datasets, this won't help.

**Partial Dependence Plots**

```python
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

# get_some_data is defined in hidden cell above.
X, y = get_some_data()
# scikit-learn originally implemented partial dependence plots only for Gradient Boosting models
# this was due to an implementation detail, and a future release will support all model types.
my_model = GradientBoostingRegressor()
# fit the model as usual
my_model.fit(X, y)
# Here we make the plot
my_plots = plot_partial_dependence(my_model,       
                                   features=[0, 2], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names=['Distance', 'Landsize', 'BuildingArea'], # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis

```


**Pipelines**:

pipelines are used for CleanerCode, Fewer Bugs, Easier to Deploy, More options for testing Models.

```python

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer

# Read Data
data = pd.read_csv('../input/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price
train_X, test_X, train_y, test_y = train_test_split(X, y)

my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)

```
Transformers are for pre-processing before modeling. The Imputer class (for filling in missing values) is an example of a transformer. Over time, you will learn many more transformers, and you will frequently use multiple transformers sequentially.

Models are used to make predictions. You will usually preprocess your data (with transformers) before putting it in a model.

You can tell if an object is a transformer or a model by how you apply it. After fitting a transformer, you apply it with the transform command. After fitting a model, you apply it with the predict command. Your pipeline must start with transformer steps and end with a model. This is what you'd want anyway.


**Cross Validation**:

In cross-validation, we run our modeling process on different subsets of the data to get multiple measures of model quality.

For example, we could have 5 folds or experiments. 
We divide the data into 5 pieces, each being 20% of the full dataset.

⟵––––––––––––––––––––––––––––Total DataSet––––––––––––––––⟶

Experiment 1 ■■■■■■■■■ □□□□□□□□□ □□□□□□□□□ □□□□□□□□□ □□□□□□□□□

Experiment 2 □□□□□□□□□ ■■■■■■■■■ □□□□□□□□□ □□□□□□□□□ □□□□□□□□□

Experiment 3 □□□□□□□□□ □□□□□□□□□ ■■■■■■■■■ □□□□□□□□□ □□□□□□□□□

Experiment 4 □□□□□□□□□ □□□□□□□□□ □□□□□□□□□ ■■■■■■■■■ □□□□□□□□□

Experiment 5 □□□□□□□□□ □□□□□□□□□ □□□□□□□□□ □□□□□□□□□ ■■■■■■■■■ 

□□□□□□□□□ Training

■■■■■■■■■ Validation

*Trade-offs Between Cross-Validation and Train-Test Split*:

For small data set Cross-Validation is necessary, for large data set Train-Test-Split is sufficient.

There's no simple threshold for what constitutes a large vs small dataset. 
If your model takes a couple minute or less to run, it's probably worth switching to cross-validation. 
If your model takes much longer to run, cross-validation may slow down your workflow more than it's worth.

Alternatively, you can run cross-validation and see if the scores for each experiment seem close. 
If each experiment gives the same results, train-test split is probably sufficient.

```python

import pandas as pd
data = pd.read_csv('../input/melb_data.csv')
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]
y = data.Price

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())

from sklearn.model_selection import cross_val_score
scores = cross_val_score(my_pipeline, X, y, scoring='neg_mean_absolute_error')
print(scores)

#You may notice that we specified an argument for scoring. 
#This specifies what measure of model quality to report. 
#The docs for scikit-learn show a list of options.

#It is a little surprising that we specify negative mean absolute error in this case. 
#Scikit-learn has a convention where all metrics are defined so a high number is better. 
#Using negatives here allows them to be consistent with that convention, though negative MAE is almost unheard of elsewhere.

#You typically want a single measure of model quality to compare between models. 
#So we take the average across experiments.

print('Mean Absolute Error %2f' %(-1 * scores.mean()))

```
*Exercise*:
Convert the code for your on-going project over from train-test split to cross-validation. 
Make sure to remove all code that divides your dataset into training and testing datasets. 
Leaving code you don't need any more would be sloppy.

Add or remove a predictor from your models. 
See the cross-validation score using both sets of predictors, and see how you can compare the scores.

**Data Lekage**:
There are two main types of leakage: *Leaky Predictors* and a *Leaky Validation Strategies*

To prevent this type of data leakage, any variable updated (or created) after the target value is realized should be excluded. 
Because when we use this model to make new predictions, that data won't be available to the model.

Data leakage can be multi-million dollar mistake in many data science applications. 
Careful separation of training and validation data is a first step, and pipelines can help implement this separation. 
Leaking predictors are a more frequent issue, and leaking predictors are harder to track down. 
A combination of caution, common sense and data exploration can help identify leaking predictors so you remove them from your model.
