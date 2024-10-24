# Optimizing-Airbnb-Pricing-Strategy-in-New-York-City-for-Hosts

# Introduction:  
Our analysis delves into the NYC Airbnb dataset, aiming to uncover underlying patterns, correlations, and the pivotal factors influencing prices, occupancy rates, and customer satisfaction. We've meticulously chosen a subset of features to examine, with a primary focus on predicting Airbnb listing prices based on these selected attributes. In addition to this predictive analysis, we'll assess how different machine learning models perform on 
this dataset.  

# Dataset:
We used ABC_NYC_2019 dataset which is available on [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)  

# Significance:  
Optimizing Airbnb pricing in NYC is significant for several reasons –  
• NYC hosts can significantly increase their income by optimizing the pricing, contributing to their financial well-being.  
• Guest Satisfaction: Accurate pricing ensures guests get the best value for their stay, leading to positive reviews and customer loyalty.  
• A more efficient pricing strategy benefits both hosts and guests, creating a win-win scenario.  
• Accurate pricing helps hosts remain competitive in the market, attracting more bookings.  

# Data Analysis:  
# 1.1 Distribution of room types across NYC:  
![image](https://github.com/user-attachments/assets/fea1c533-fa0d-4291-90d6-c4e5f1e2e754)  

# 1.2 Analyzing Room type by Neighborhood groups:  
![image](https://github.com/user-attachments/assets/39752481-5c92-49e0-a163-a7d029f2e655)  

# 1.3 Mean Price by Neighborhood groups:  
![image](https://github.com/user-attachments/assets/25265c6c-93f4-4fd6-afae-b8a54045a2b4)  

# 1.4 Minimum Nights vs Price:  
![image](https://github.com/user-attachments/assets/7b1e1abc-72f1-4a9b-880f-bf9ce55efb9d)  

# Machine Learning Algorithms used:  
1. Linear Regression
2. Naive Bayes Classifier
3. Decision Tree Regressor
4. XGBoost Regressor
5. Support Vector Regressor
6. K-Nearest Neighbor Regressor
7. Ensemble Random Forest
8. Neural Networks
   
We tried different models and it is evident that XGBoost outperformed others for this data.  

![image](https://github.com/user-attachments/assets/fadeab6c-74f9-4de2-8281-af99921f23b1)


While XGBoost demonstrates good performance, the potential for even better results exists with a more robust dataset. Given that regression tasks performance evaluation cannot be relied on accuracy metric, our focus primarily lied on improving the Root Mean Squared Error (RMSE) and R^2 values. This improvement can be achieved by augmenting the dataset significantly and incorporating business insights to better understand the factors impacting prices. For better performance, we have also tried to preprocess data in various ways to see if it will perform better, changed the features that we considered for training but the results did not seem to change much and got even worse in few cases. The correlation matrix for the current dataset reveals weak correlations between the features and the target variable (as illustrated below), indicating potential issues with the existing dataset.  
![image](https://github.com/user-attachments/assets/10172f15-2e2f-4348-b01d-37c7c4dc4a20)


