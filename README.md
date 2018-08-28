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
    ml_model = DecisionTreeRegressor(random_state=0)

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
    print(mean_absolute_error(val_y, val_predictions))
```