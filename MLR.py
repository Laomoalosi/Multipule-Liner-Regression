# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:29:24 2019

@author: Laone Moalosi
"""

import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

# Load the dataset into a Dataframe.
df = pd.read_csv("FyntraCustomerData.csv", na_values = ["?"])

# Replace any missing values in the dataset with the most frequently occurring data in each column
new_df = df.fillna({'Avg_Session_Length': df.Avg_Session_Length.mode(),
                    'Time_on_App': df.Time_on_App.mode(),
                        'Time_on_Website': df.Time_on_Website.mode(),
                            'Length_of_Membership': df.Length_of_Membership.mode(),
                                'Yearly_Amount_Spent': df.Yearly_Amount_Spent.mode()})

print(new_df)

# 3. No Correlation
sns.jointplot(x="Time_on_Website", y="Yearly_Amount_Spent", data=new_df)

# 4. Positive Correlation 
sns.jointplot(x="Time_on_App", y="Yearly_Amount_Spent", data=new_df)

# 5. Length of Membership looks to be the most correlated feature with Yearly Amount Spent
sns.pairplot(new_df)





# To check number of missing values in each column
# print(new_df.isnull())

"""
# 6.linear model plot for Length of Membership and Yearly Amount Spent
X = new_df.iloc[:, 6].values.reshape(-1, 1)  # values converts it into a numpy array
Y = new_df.iloc[:, 7].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.title('Test Data')
plt.xlabel('Length of Membership')
plt.ylabel('Yearly Amount Spent')
plt.show()
"""


