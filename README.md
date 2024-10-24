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

More Detailed Analysis - [Phase 1](https://github.com/alla-sahithya/Optimizing-Airbnb-Pricing-Strategy-in-New-York-City-for-Hosts/blob/main/phase%201/alla16_marziyek_report_phase_1.pdf)

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

More Detailed Analysis - [Phase 2](https://github.com/alla-sahithya/Optimizing-Airbnb-Pricing-Strategy-in-New-York-City-for-Hosts/blob/main/phase%202/alla16_marizyek_phase2_report.pdf)

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

# Neighbourhood Prices Page:  

![image](https://github.com/user-attachments/assets/eacb1e64-1322-420d-8e77-0b6206274a38)  

![image](https://github.com/user-attachments/assets/b23e0437-1581-4353-a9e5-c03cd2eb5b87)


# Visualizations Page:  

![image](https://github.com/user-attachments/assets/edeb5c25-5363-4616-a1e7-fb12dc69cd54)  

![image](https://github.com/user-attachments/assets/72d81f85-9273-428b-b140-37097045526b)  

![image](https://github.com/user-attachments/assets/796f621b-4277-499a-8a82-6b7bcb5a1775)  

![image](https://github.com/user-attachments/assets/90b356f3-5565-4352-b686-2fa6189e50f8)  

# Future Scope:  
• The analysis highlights the need for a more robust and accurate dataset to achieve better price predictions. It is also recognized that the current dataset reveals weak correlations between features and the target variable, indicating issues with the existing data. Hence, collecting additional data points related to Airbnb listings which also include information about property features, amenities, nearby local attractions, and guest reviews will help us get a more robust and reliable product. This will also help the hosts to understand user preferences.  
• To further enhance model performance, future work should also focus on incorporating business insights, and exploring more advanced feature engineering techniques.  
• Also, we can implement algorithms that adapt to the real-time market fluctuations, seasonal trends, and special events which will empower hosts to optimize their pricing strategies. Adjusting prices based on demand and supply dynamics will help the hosts to maximize their revenue potential while staying competitive in the market.  
• Future scope can also be leveraging geospatial data which can provide valuable insights into the influence of location on Airbnb prices.  

# References:  
1. [https://docs.streamlit.io/](https://docs.streamlit.io/)
2. [https://matplotlib.org/stable/gallery/index.html](https://matplotlib.org/stable/gallery/index.html)
3. [https://seaborn.pydata.org/](https://seaborn.pydata.org/)
4. [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/)
5. [https://matplotlib.org/stable/](https://matplotlib.org/stable/)
6. [https://www.simplilearn.com/10-algorithms-machine-learning-engineers-need-to-know-article](https://www.simplilearn.com/10-algorithms-machine-learning-engineers-need-to-know-article)
7. [https://www.datacamp.com/tutorial/xgboost-in-python](https://www.datacamp.com/tutorial/xgboost-in-python)
8. [https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regressionmodel-418ca481755](https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regressionmodel-418ca481755)
9. [https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)




