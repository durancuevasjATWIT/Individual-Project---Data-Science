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
  1-Sklearn: Machline learning library used for creating test and train sets
  2-Pandas + Numpy: Data manipulation
  3-Seaborn + Matplotlib: Data visualization
  4-Jupyter: Development notebook
  5-GitHub: Version control

   ### Methods used with scikit:

   ### Features

   ### Pipeline

## Result

  #### Q1 - Predicting Game Ratings:
  The exploration of various feature engineering techniques and regression algorithms yielded compelling results for predicting game ratings. Random Forest Regression, 
  onsistently achieved impressive training accuracies ranging between 50% to 90%. The strategic deployment of this model resulted in a test accuracy of 55%, affirming its 
  effectiveness in accurately predicting game ratings.

  #### Q2 - Categorizing Games into Genres:
  In the pursuit of categorizing games into genres, the application of classification algorithms (Random Forest Classification) proved fruitful. The deployed models
  exhibited robust performance, showcasing the ability to accurately classify games into their respective genres.

  #### Q3 - Predicting Game Plays:
  The endeavor to forecast the number of plays a game would receive involved the application of Random Forest Regression. Notably, the Random Forest Regression model
  demonstrated effectiveness in predicting numerical values, showcasing its utility in forecasting game plays. The results were validated through rigorous testing, 
  roviding insights into the potential success of this predictive model.

  #### Overall Performance:
  The overall success of the deployed models highlight the effectiveness of the chosen techniques and algorithms in addressing the specific research questions.


## Discussion
In my exploration of game analytics, I found success in using different techniques and algorithms to enhance predictions. Notably, Random Forest Regression consistently achieved high training accuracies, even more than Linear Regression ranging from 50% to 90%. Encouraged by this performance, I made a Random Forest Regression model, aiming for efficiency and accuracy in predicting game ratings, categorizing genres, and forecasting play counts. The model's effectiveness was confirmed with a test accuracy ranging from 50% to 90% as stated before.

Digging into the details of the training data, I performed extensive analysis to understand why it was not easy to achieve higher accuracy, which led me to having to test different models while trying different types of techniques to see which one gave the best results. This also challenged me to delve deeper into the relationships between different features, looking for nuanced information to improve predictions.

## Summary
In summary, my exploration into game analytics centered around achieving accurate predictions using various techniques and algorithms. Notably, Random Forest Regression consistently demonstrated strong training accuracies, ranging between 50% and 90%. This success led to the strategic deployment of a well-crafted Random Forest Regression model, prioritizing efficiency and accuracy in predicting game ratings, categorizing genres, and forecasting play counts. The model's effectiveness was affirmed by a commendable test accuracy of 50% - 90%.

## References

- [[Kaggle](https://www.kaggle.com/datasets/arnabchaki/popular-video-games-1980-2023)].
