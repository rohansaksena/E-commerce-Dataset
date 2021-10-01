# E-commerce Dataset
This is my first project involving Machine Learning. We perform Linear Regression on our dataset to make some predicitons.
We will work on a E-commerce Dataset from a company. It has customer info such as Email, Address, Avatar, Avg. Session Length, Time on App, Time on Website, Length of Membership, Yearly Amount Spent. We will perform Exploratory Data Analysis, Data Visualisation and then perform some Predictions.

## Source of data: 
Kaggle 

## Project Outcomes:
by the end of this project we would have sucessfully answered a few questions :

### *The Imports*
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### Reading the Data as customers
```
customers= pd.read_csv("Ecommerce Customers")
```

### Basic Info of our Data
```
customers.head()

customers.info()

customers.describe()
```

### *Lets Explore..!*

### Create a jointplot to compare the time on app and the yearly amount spent columns
```
sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=customers,kind='reg')
```

### Create a jointplot to compare the time on website and the yearly amount spent columns
```
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers,kind='reg')
```

### Use Seaborn to compare time on app and length of membership
```
sns.jointplot(x='Time on App',y='Length of Membership',data=customers,kind='hex')
```

### Create a pairplot to visualise the entire dataset
```
sns.pairplot(customers)
```

### Create a lmplot for Yearly Amount Spent vs Length of Membership
```
sns.lmplot(x='Yearly Amount Spent',y='Length of Membership',data=customers)
```

### *Data Predictions*
#### Let's move forward to Predict Some Data

### Divide the data into Training and Testing sets
```
customers.columns

y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
```

### Use cross validation train test split from sklearn to split the data into training and testing data sets
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
```

### Training the model using linear regression
```
from sklearn.linear_model import LinearRegression
```

### Create an object of LinearRegression()
```
lm =LinearRegression()
```

### Train/fit your model
```
lm.fit(X_train, y_train)
```

### Print out the coefficients of the model
```
lm.coef_
```

### *Predicting Test Data*

### Now that we have fit our model lets evaluate its performance by predicting off the test values
```
predictions = lm.predict(X_test)
```

### Plotting a scatterplot for real test values vs the predicted values
```
sns.scatterplot(x=y_test,y=predictions)
plt.grid()
plt.ylabel('True Values')
```

### *Evaluate the model*

### Calculate the mean absolute error, mean squared error, and the root mean square error
```
from sklearn import metrics
print("MAE :",metrics.mean_absolute_error(y_test,predictions))
print("MSE :",metrics.mean_squared_error(y_test,predictions))
print("RMSE :",np.sqrt(metrics.mean_squared_error(y_test,predictions)))
```

### Explained Variance Score
```
metrics.explained_variance_score(y_test,predictions)
```

### Residuals

### Plot a histogram of the residuals to make sure it looks normally distributed
```
sns.displot(y_test-predictions,kde=True,bins=50,lw=2,edgecolor='white',color='green')
```

### *Conclusion*

### We need an answer to the question of whether should the company focus on there mobile app or should they focus on web development
```
pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
```

## Project Setup:
To clone this repository you need to have Python compiler installed on your system alongside pandas and seaborn libraries. I would rather suggest that you download jupyter notebook if you've not already.

To access all of the files I recommend you fork this repo and then clone it locally. Instructions on how to do this can be found here: https://help.github.com/en/github/getting-started-with-github/fork-a-repo

The other option is to click the green "clone or download" button and then click "Download ZIP". You then should extract all of the files to the location you want to edit your code.

Installing Jupyter Notebook: https://jupyter.readthedocs.io/en/latest/install.html<br>
Installing Pandas library: https://pandas.pydata.org/pandas-docs/stable/install.html


































