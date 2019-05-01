# NYC-Taxi-fare-prediction

The main objective of this project is to predict the fare amount of the taxi in NYC. The dataset is taken from Kaggle. The data cleaning process
is done using R technology and stored the cleaned data in new CSV file. This CSV was imported in Tableau for visualization.
Being a multivariate dataset,before applying for regression we tried to reduce the dimensionality of dataset using PCA but it failed.
So after using stepwise AIC we got the final predictors and then applied various algorithms.
Implemented algorithms like Multiple linear regression, Random Forest regressor and XGBoost for predicting the fare amount based on various 
predictor variables.
