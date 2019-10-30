# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:29:24 2019

@author: Laone Moalosi
"""


import pandas as pd  # To read data
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt  # To visualize  # from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 1. Load the dataset into a Dataframe.
df = pd.read_csv("FyntraCustomerData.csv", na_values = ["?"])

# 2. Replace any missing values in the dataset with the most frequently occurring data in each column
new_df = df.fillna({'Avg_Session_Length': df.Avg_Session_Length.mean(),
                    'Time_on_App': df.Time_on_App.mode(),
                    'Time_on_Website': df.Time_on_Website.mean(),
                    'Length_of_Membership': df.Length_of_Membership.mode(),
                    'Yearly_Amount_Spent': df.Yearly_Amount_Spent.mode()})

print(new_df)


# 3. No Correlation
# sns.jointplot(x="Time_on_Website", y="Yearly_Amount_Spent", data=new_df)

# 4. Positive Correlation 
# sns.jointplot(x="Time_on_App", y="Yearly_Amount_Spent", data=new_df)

# 5. Length of Membership looks to be the most correlated feature with Yearly Amount Spent
# sns.pairplot(new_df)


"""
# 6.linear model plot for Length of Membership and Yearly Amount Spent. YES it fits well
X = new_df.iloc[:, 6].values.reshape(-1, 1)  # values converts it into a numpy array
Y = new_df.iloc[:, 7].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions


plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.title('Test Data')
plt.xlabel('Length_of_Membership')
plt.ylabel('Yearly_Amount_Spent')
plt.show()
"""

"""

# 6. Create a linear model plot for Length of Membership and Yearly Amount Spent YES it fits well
sns.lmplot(x='Length_of_Membership', y='Yearly_Amount_Spent', data=new_df)

"""

# Train and Test the data
y = new_df['Yearly_Amount_Spent']
X = new_df[['Avg_Session_Length', 'Time_on_App', 'Time_on_Website', 'Length_of_Membership']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


model = LinearRegression()
model.fit(X_train, y_train)

print('Coefficients: \n', model.coef_)



# 7. Random state is used for initializing the internal random number generator, 
#    which will decide the splitting of data into train and test indices in your case


# 8. Predict the data and do a scatter plot. The actual data and predicted values match quite a lot except for a few points
Y_Predict= model.predict(X_test)

plt.scatter(y_test, Y_Predict)
plt.xlabel('actual values')
plt.ylabel('predicted values')
plt.show()

# 9.Root Mean Squared Error (RMSE) 
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_Predict)))

# 10. They should focus more on their website 


# To check number of missing values in each column
# print(new_df.isnull().sum())

# shows the missing records
# print(df[df.Avg_Session_Length.isnull()])
