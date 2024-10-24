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
   
We experimented with various machine learning algorithms and we chose XGBoost model for further analysis of our project. From the below summary of metrics, it is evident that XGBoost outperformed others for this data.  

![image](https://github.com/user-attachments/assets/fadeab6c-74f9-4de2-8281-af99921f23b1)


While XGBoost demonstrates good performance, the potential for even better results exists with a more robust dataset. Given that regression tasks performance evaluation cannot be relied on accuracy metric, our focus primarily lied on improving the Root Mean Squared Error (RMSE) and R^2 values. This improvement can be achieved by augmenting the dataset significantly and incorporating business insights to better understand the factors impacting prices. For better performance, we have also tried to preprocess data in various ways to see if it will perform better, changed the features that we considered for training but the results did not seem to change much and got even worse in few cases. The correlation matrix for the current dataset reveals weak correlations between the features and the target variable (as illustrated below), indicating potential issues with the existing dataset.  
![image](https://github.com/user-attachments/assets/10172f15-2e2f-4348-b01d-37c7c4dc4a20)

# Deployment:  
We have created a Streamlit web application with different pages for home, predicting prices, neighbourhood prices, and visualizations. We also used session state to store input features and predicted prices.  
# Pages in Streamlit App:
- **Home Page:** Displays a welcome message.
- **Predict Price Page:** Allows users to input features and predicts the Airbnb price. Provides feedback and insights.
- **Neighbourhood Prices Page:** Displays top 5 listings based on user-selected neighbourhood and room type. Allows exploration of prices across different neighborhood groups and room types.
- **Visualizations Page:** Presents visualizations like price distribution, top hosts, and average prices of different room types based on user-selected neighbourhood groups, distribution of room types 
in NYC, demand for room types according to their prices.
**Styling:** Applied custom styling for the Streamlit app, including a coloured background for the title.

# Instructions to set up and run the code:
1. Setup:  
• Install required packages: ‘streamlit’, ‘pandas’, ‘xgboost’, ‘matplotlib’, ‘seaborn’, ‘scikit-learn’, ‘plotly’.  
• Also, confirm if you have the ‘ABC_NYC_2019.sv’ dataset. In case you want to use a new data set, you just have to change the dataset name in line 18 of the code in custom.py file (you can open the file from vs code to edit, if required).   
• Make sure that you have all the necessary columns.  

2. Running the App:  
• Run the Streamlit app using the command: ‘streamlit run custom.py’   
• You can run the above command in the terminal by opening the terminal from the location where you have your custom.py  
• Then automatically a local host page will be opened. In case it does not open by itself, you can click or open the Local host URL that will be shown on your terminal.

3. Navigation:  
• Use the side bar to navigate between different pages (Home, Predict Price, Neighbourhood Prices, and Visualizations).  
• You can also follow the on-screen instructions for predicting prices and exploring neighbourhood prices.

# User Interface:  
# Home Page and Drop down options:  
![image](https://github.com/user-attachments/assets/bd6a522f-b8af-4bbd-b995-b5dbf4606d87)  

![image](https://github.com/user-attachments/assets/a2359d19-9a94-4552-b45d-0c5edcf51e0d)  

# Predict Price Page:  

![image](https://github.com/user-attachments/assets/cfaee543-a7d5-4f08-9ef7-ee61361259bf)  

![image](https://github.com/user-attachments/assets/490b1190-1eb4-406c-b3dc-f124cba395f1)


