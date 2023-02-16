# Timeseries Framework (TSFW)
A timeseries framework that iterates over and fits various different models and outputs the result in the form of plots and accuracy (MAPE). It performs different feature combinations for different model classes and then outputs a table that reflects


## How to use

The lib.py file has all the relevant classes and functions that are used in the demo.ipynb file. To run the demo.ipynb file with your own timeseries dataset, make sure that your dataset is a csv file with at least a date column (dates should not repeat i.e you can't have mutiple entries with the same date) and a target column. 


### Models supported
1. LSTMs (needs some working to get better results)
2. Prophet
3. Linear Regression

### Features
1. Perform feature combinations and select best model for each model class
2. Generate datetime features (holidays included: only works for time series that is on a daily level)
3. Generate lag features (for Linear Regression)
4. Basic data pre-processing
5. Feature importance plots and prediction plots (need to incorporate SHAP for LSTMs)

### Packages needed
To run this, you mainly need the following packages
1. numpy
2. pandas
3. matplotlib
4. sklearn
5. holidays
6. keras
7. prophet
8. tensorflow
