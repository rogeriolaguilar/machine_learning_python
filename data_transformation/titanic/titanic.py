import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

# Improving data visualization
np.set_printoptions(threshold=None, precision=2) # reduce precision

# Read titanic base from Kaggle: https://www.kaggle.com/c/titanic/data
titanic_train = pd.read_csv('data/train.csv')

print('Dimention:', titanic_train.shape)
print('Keys:',titanic_train.keys())
print('Types:',titanic_train.dtypes)


print('Descriptive statistics:')
print(titanic_train.describe())


# print('Non numeric attributes (object)')

print("Object dtypes")
non_numeric_categories = titanic_train.dtypes[titanic_train.dtypes == "object"]
print(non_numeric_categories)

# print("Showing how boolean indexing happens")
# boolean_series = titanic_train.dtypes == "object"
# print(boolean_series)
# non_numeric_categories = titanic_train.dtypes[titanic_train.dtypes == "object"]
# print(non_numeric_categories)

print(">> Make `Survived` a categorical attribute")
print("Before \n", titanic_train["Survived"].describe())
categoric_survived = pd.Categorical(titanic_train['Survived']).rename_categories(["Died", "Survived"])
print("Categorical:")
print(categoric_survived.describe())
print(type(categoric_survived))

titanic_train["Survived"] = categoric_survived

print(">> Make classes categorical")
print(titanic_train["Pclass"].describe)
categoric_pclass = pd.Categorical(titanic_train['Pclass'], 
                                  ordered=True).rename_categories(["1a-class",
                                                                   "2a-class", 
                                                                   "3a-class"])
print("Categorical")
print(categoric_pclass.describe())

titanic_train['Pclass'] = categoric_pclass 


print("Remove unused fields")
del titanic_train['PassengerId']
del titanic_train['Ticket']


print(">> Checking cabin field")
print(titanic_train["Cabin"].unique())

print("The first letter represents the floor, which can be useful to the categorization")
cabins = titanic_train["Cabin"].astype(str)
floor_letters=[cabin[0] for cabin in cabins]

categorical_cabin_letter=pd.Categorical(floor_letters)


print("Categoric floor letters")
print(categorical_cabin_letter.describe())

titanic_train["Cabin"] = categorical_cabin_letter


print("Analyzing age column")

result = titanic_train.hist(column='Age',
                            bins=20) # Columns


age=titanic_train['Age'].dropna() # Drop nan values
print("Age: \n", age.unique())

age_median=np.median(age)
print("Age median of the existing elements:", age_median)


print("Add the median whe age is null")
new_age = np.where(titanic_train['Age'].isnull(), age_median, titanic_train["Age"])

titanic_train["Age"]=new_age

print("New age", titanic_train["Age"].describe())

titanic_train.hist(column='Age',
                            bins=20) # Columns
plt.show(block=False) 


print(">> Analyse Fares")
print(titanic_train["Fare"].describe()) # The max value is much hiegher than the third quartile

fig, ax = plt.subplots() # Make it create in a new figure
titanic_train["Fare"].plot(kind="box", ax=ax)


index = np.where(titanic_train["Fare"] == max(titanic_train["Fare"]) )
print("Extreme values:",titanic_train.loc[index], sep='\n')


print("Create new attr Family")
print(titanic_train["SibSp"].describe())
print(titanic_train["Parch"].describe())


titanic_train["Family"] = titanic_train["SibSp"] + titanic_train["Parch"]
#print(titanic_train["Family"])


print("Check correlation")
int_fields = titanic_train.dtypes[titanic_train.dtypes == "int64"].index
corr = np.corrcoef(titanic_train[int_fields].transpose())
correlation = pd.DataFrame(data=corr, index=int_fields, columns=int_fields)


# print("Formatted training base")
# print(titanic_train.dtypes)


# print(type(titanic_train))

plt.show() # show all graphics
