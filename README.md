# Predicting-Video-Game-Sales
Project: Predicting Video Game Sales video game project

Description: 

The global video game industry is one of the most dynamic and rapidly growing sectors in entertainment, generating over $180 billion in annual revenue — surpassing both the film and music industries combined. This expansive market includes blockbuster console titles, indie PC games, and mobile games downloaded by millions. Consumer interest is driven by diverse factors such as game genre, critical reviews, platform availability, developer reputation, release timing, and marketing effectiveness. With thousands of titles released each year across multiple platforms, understanding the elements that influence game sales is crucial for developers, publishers, and marketers alike.

Objective:

The goal of this project is to apply machine learning techniques to predict global sales (in millions of units) of video games using a provided dataset. You will engage in the full predictive modeling pipeline, including data preprocessing, feature engineering, model training, and performance evaluation.

Dataset:

Download final_test.csv
The training set (final_train.csv) is the primary resource for building your machine learning models. In this set, each record includes the actual sales (variable name: 'total_sales') for a given video game, which serves as the ground truth or "label" for your model. The dataset includes attributes such as platform, release year, genre, publisher, and regional sales figures. There are 11,894 observations in the training set. You are encouraged to explore the data, generate new features, apply domain knowledge, or even merging in additional information from other dataset to improve your model's performance.

The test set (final_test.csv) have 7028 rows. It includes similar features but omits the total_sales column. Your task is to predict the expected sales figures (in millions of units sold) for each game in this test set. There are no restrictions on which modeling techniques you may use — feel free to experiment with linear regression, tree-based models, ensemble methods, or any other suitable approach

Evaluation
Goal is to predict the market price for each car in the test set. Try your best to maximize the coefficient of determination ( i.e.,  R-squared) of your model.  The  provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model. 
