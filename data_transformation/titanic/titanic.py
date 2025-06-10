import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

# Improving data visualization
np.set_printoptions(threshold=None, precision=2) # reduce precision
pd.set_option('display.precision', 2) 
pd.set_option('display.max_columns', 500) # ensure all columns are displayed
pd.set_option('display.max_rows', 500) # ensure all lines are displayed


titanic_train = pd.read_csv('data/train.csv')

print('Dimention:', titanic_train.shape)
print('Keys:',titanic_train.keys())
print('Types:',titanic_train.dtypes)


print('Descriptive statistics:')
print(titanic_train.describe())


print('Non numeric attributes (object)')

# Filter non object dtypes by using boolean indexing
non_numeric_categories = titanic_train.dtypes[titanic_train.dtypes == "object"]
print(non_numeric_categories)

# print("Showing how boolean indexing happens")
# boolean_series = titanic_train.dtypes == "object"
# print(boolean_series)
# non_numeric_categories = titanic_train.dtypes[titanic_train.dtypes == "object"]
# print(non_numeric_categories)
