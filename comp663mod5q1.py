''' Logistic Regression using logit
Using the csv file nbaallelo.csv and the logit function, construct a
logistic regression model to classify whether a team will win or lose
a game based on the team's elo_i score.

Read in the file nbaaello.csv.
The target feature will be converted from string to a binary feature by the provided code.
Split the data into 70 percent training set and 30 percent testing set. Set random_state = 0.
Use the logit function to construct a logistic regression model with wins as the target
and elo_i as the predictor.
Print the coefficients of the model.'''

# import the necessary libraries
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

# load nbaallelo.csv into a dataframe
df = pd.read_csv('nbaallelo.csv')# code to load csv file


# Converts the feature "game_result" to a binary feature and adds as new column "wins"
wins = df.game_result == "W"
bool_val = np.multiply(wins, 1)
wins = pd.DataFrame(bool_val, columns = ["game_result"])
wins_new = wins.rename(columns = {"game_result": "wins"})
df_final = pd.concat([df, wins_new], axis=1)

print(df_final.head())

# split the data df_final into training and test sets with a test size of 0.3 and random_state = 0
train, test = train_test_split(df_final, test_size=0.3, random_state=0) # code to split df_final into training and test sets

# construct a logistic model with wins and the target and elo_i as the predictor, using the training set
lm = smf.logit(formula = 'wins ~ elo_i', data = train).fit()# code to construct logistic model using the logit function

# print coefficients for the model
print(lm.params) # code to return coefficients
