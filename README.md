# Individual Project - Data Science

## Introduction
The motivation and objective behind this comprehensive project was to delve and explore into the analysis of a dataset comprising various game-related attributes. The primary research questions focused on predicting game ratings, categorizing games into genres, and forecasting the number of plays a game might receive. These questions formed the tested hypotheses, driving the purpose of the research which use different types of models as well as some tools which are going to be explained below.

## Data Selection
The dataset used for this project contains information about various games, including features such as genre, platform, release year, and more. The data is sourced from [[Kaggle](https://www.kaggle.com/datasets/arnabchaki/popular-video-games-1980-2023)].

Data Preview
![image](https://github.com/durancuevasjATWIT/Individual-Project---Data-Science/assets/90558252/76b8da6a-631e-4c23-a4ea-bbbc545684f8)

The dataset encompassed over a thousand samples, reflecting a mix of categorical and numerical variables. The decision to employ this dataset was motivated by its relevance to gaming analytics and the availability of features conducive to machine learning exploration.

Data preprocessing played a pivotal role in ensuring the dataset's suitability for modeling. Categorical variables underwent encoding using OneHotEncoder, ensuring their compatibility with machine learning algorithms. StandardScaler was applied to facilitate feature scaling, and SimpleImputer which addressed missing values.


## Methods
  ### Tools
 - NumPy, SciPy, Pandas for data analysis
 - Scikit-learn for Machine Learning
 - Jupyter as IDE

   ### Methods used with scikit:

   ### Features

   ### Pipeline

## Result

## Discussion
In my exploration of game analytics, I found success in using different techniques and algorithms to enhance predictions. Notably, Random Forest Regression consistently achieved high training accuracies, even more than Linear Regression ranging from 50% to 90%. Encouraged by this performance, I made a Random Forest Regression model, aiming for efficiency and accuracy in predicting game ratings, categorizing genres, and forecasting play counts. The model's effectiveness was confirmed with a test accuracy ranging from 50% to 90% as stated before.

Digging into the details of the training data, I performed extensive analysis to understand why it was not easy to achieve higher accuracy, which led me to having to test different models while trying different types of techniques to see which one gave the best results. This also challenged me to delve deeper into the relationships between different features, looking for nuanced information to improve predictions.

## Summary
