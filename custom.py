# importing required libraries
import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# loading Airbnb dataset
data = pd.read_csv('AB_NYC_2019.csv')

stop_words = stopwords.words('english')

def clean_text_data(sentence, stop_words):
    sentence = sentence.lower()  # converting the text into lower case to standardize the text
    sentence = sentence.split()
    text = ""
    for word in sentence:  # removing stopwords from the sentence
        if word in stop_words:
            continue
        else:
            text = text + word + " "
    text = text.strip()
    punctuations = string.punctuation
    text = ''.join([punc for punc in text if punc not in punctuations])  # removing punctuations from the sentence
    # there are different symbols like ❥, çº½çº¦ä¹‹å®¶ in the data, hence to remove them, we are just considering the text that is in A-Z, a-z or digits
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text) 
    text = ' '.join(text.split())
    return text


# removing outliers using iqr technique
def remove_outliers_using_iqr(df, col_name):
    count = 0
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3-q1
    lowerbound = q1 - 1.5 * iqr
    upperbound = q3 + 1.5* iqr
    outliers = (df[col_name] < lowerbound) | (df[col_name] > upperbound)
    df = df[~outliers]
    return df
    
def cleaning_data(df, stop_words):
    df['name'].fillna('Not known', inplace = True)
    df['host_name'].fillna('Not known', inplace = True)
    df['reviews_per_month'].fillna(0, inplace = True)
    df.drop(['last_review'], axis = 1, inplace = True)
    df.drop_duplicates(inplace=True) 
    df['name'] = df['name'].apply(lambda x: clean_text_data(str(x),stop_words))  # cleaning text data in name column
    df['host_name'] = df['host_name'].apply(lambda x: str(x).lower())  # lowercasing all names in host_name column
    df["room_type"] = df["room_type"].str.lower().str.replace('/',' or ')  # normalizing text data in room type column (to replace special character '/' with or)
    df['neighbourhood_group'] = df['neighbourhood_group'].apply(lambda x: clean_text_data(str(x),stop_words))  # cleaning - lowercasing, punctuations
    df['neighbourhood'] = df['neighbourhood'].str.lower()  # lowercasing all neighbourhood data
    # As price cannot be 0 if you want to rent out a property, if price = 0 for any listing, it is considered as anamoly. Hence remove the rows with price=0
    df = df[~(df['price'] == 0)]  # removing listings with 0 price values 
    numerical_columns = ['price', 'minimum_nights','number_of_reviews', 'availability_365']  # we are removing outliers from all numerical column so that our model should not be 
    for col in numerical_columns:
        df = remove_outliers_using_iqr(df, col)
    return df

data = cleaning_data(data, stop_words)  # calling clean data function
# captializing first letter of the word 
data['neighbourhood'] = data['neighbourhood'].str.capitalize()  
data['neighbourhood_group'] = data['neighbourhood_group'].str.capitalize()
data['room_type'] = data['room_type'].str.capitalize()

numerical_columns = ['minimum_nights', 'number_of_reviews', 'availability_365']
categorical_columns = ['neighbourhood_group', 'room_type', 'neighbourhood']

# features and target variable
X = data[['latitude', 'longitude'] + numerical_columns + categorical_columns]
y = data['price']


# encoding categorical variables
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# splitting the data into training and testing sets (70-30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# training the XGBoost model
model = xgb.XGBRegressor(n_estimators=145, learning_rate=0.1, colsample_bytree=0.8, gamma=0.1)
model.fit(X_train, y_train)

# function to predict price
def predict_price(features):
    features_encoded = features.copy()

    # ensuring that the order of features matches the order used during training
    ordered_columns = ['latitude', 'longitude'] + numerical_columns + categorical_columns
    features_encoded = features_encoded[ordered_columns]

    for column in categorical_columns:
        features_encoded[column] = label_encoders[column].transform([features_encoded[column]])[0]
    return model.predict(features_encoded)[0]

# initializing session state
if 'predicted_price' not in st.session_state:
    st.session_state.predicted_price = None

if 'input_features' not in st.session_state:
    st.session_state.input_features = {}

# streamlit app
st.set_page_config(page_title='Airbnb Price Prediction', page_icon=':house_with_garden:')

# applying background color to the heading
st.markdown(
    """
    <style>
        .title {
            background-color: #3080b9;
            color: white;
            padding: 0.5rem;
            border-radius: 0.25rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# heading with background color
st.write(
    f'<h1 class="title">Airbnb Price Prediction</h1>',
    unsafe_allow_html=True
)

# sidebar for navigation
selected_tab = st.sidebar.selectbox("Select Page", ["Home", "Predict Price", "Neighbourhood Prices", "Visualizations"])

# Home Page
if selected_tab == "Home":
    st.subheader('Welcome to the Airbnb Price Prediction App!')
    st.write("Explore different features and functionalities using the tabs on the left.")

# Predict Price Page
elif selected_tab == "Predict Price":
    st.subheader('Predict Price')

    # define custom names for numerical columns
    numerical_column_names = {
        'minimum_nights': 'Minimum nights required for stay',
        'number_of_reviews': 'Total reviews for the listing',
        'availability_365': 'Availability of the listing throughout the year'
    }

    # selectbox for neighbourhood group
    st.session_state.input_features['neighbourhood_group'] = st.selectbox('Neighbourhood Group', data['neighbourhood_group'].unique(), index=None, placeholder='Select One')
    # filter and selectbox for neighbourhood
    filtered_neighborhoods = data[data['neighbourhood_group'] == st.session_state.input_features['neighbourhood_group']]['neighbourhood'].unique()
    st.session_state.input_features['neighbourhood'] = st.selectbox('Neighbourhood', filtered_neighborhoods, index=None, placeholder='Select One')

    # selectbox for room type
    st.session_state.input_features['room_type'] = st.selectbox('Room Type', data['room_type'].unique(), index=None, placeholder='Select One')

    # input features for prediction
    for column in numerical_columns:
        custom_column_name = numerical_column_names.get(column, column)
        if column == 'availability_365':
            st.session_state.input_features[column] = st.number_input(f'{custom_column_name}', min_value=0, max_value=365, step=1)  # value for availability column should be less than 365
        else:
            st.session_state.input_features[column] = st.number_input(f'{custom_column_name}', step=1)

    # float input for latitude with .5f precision
    st.session_state.input_features['latitude'] = st.number_input('Latitude',step=0.0001, format="%.5f")

    # float input for longitude with .5f precision
    st.session_state.input_features['longitude'] = st.number_input('Longitude', step=0.0001, format="%.5f")

    # predict price button
    if st.button('Predict Price'):
        # checking if any required field is empty
        if any(value in ['Select One', None] for value in st.session_state.input_features.values()):
            st.error("Please select a value for all the required fields.")
        else:
            # prediction
            input_data = pd.DataFrame([st.session_state.input_features])
            st.session_state.predicted_price = predict_price(input_data)
            st.success(f'Predicted Price: ${st.session_state.predicted_price:.2f}')
        
            # feedback
            st.subheader('Feedback and Insights')
            st.write("The predicted price gives an estimate of the expected price for the Airbnb listing based on the provided features. "
                    "You can leverage this information to make informed decisions when setting the price for your own listing based on the amenities you provide.")

            st.write("You may consider exploring additional features that could influence the price of your Airbnb listing. "
                     "Factors such as proximity to popular attractions, public transportation, or local amenities could influence the price.")

            st.subheader('Visualization and Filtering')
            st.write("Explore the Neighbourhood Prices page on the left to view the prices of 5 Airbnb listings in the same neighborhood and room type. "
                     "This comparison offers quick insights into local market prices, helping you optimize your pricing strategy. "
                     "Additionally, you can select the neighbourhood group and room type to delve deeper into the price dynamics of specific areas.")

            st.write("You can also explore the visualizations page to see trends across various attributes that might help you to decide on price of your listing.")


# Neighborhood Prices Page
elif selected_tab == "Neighbourhood Prices":
    st.subheader('Neighbourhood Prices')

    # checking if necessary keys are present in input_features
    if 'neighbourhood_group' in st.session_state.input_features and 'room_type' in st.session_state.input_features:
        neighborhood = st.session_state.input_features['neighbourhood_group']
        room_type = st.session_state.input_features['room_type']
        
        # filtering and displaying nearby listings
        nearby_listings = data[(data['neighbourhood_group'] == neighborhood) & (data['room_type'] == room_type)].head(5)

        if not nearby_listings.empty:
            st.markdown('<h3 style="font-size: 1.5em;">Top 5 Airbnb Listings in the Seleted Neighborhood Group and Room Type</h3>', unsafe_allow_html=True)

            # renaming the columns before displaying the table
            nearby_listings_display = nearby_listings[['neighbourhood_group', 'neighbourhood', 'price']].reset_index(drop=True)
            nearby_listings_display = nearby_listings_display.rename(columns={
                'neighbourhood_group': 'Neighbourhood Group',
                'neighbourhood': 'Neighbourhood',
                'price': 'Price'
            })

            st.table(nearby_listings_display.style.format({'Price': "${:.2f}"}))
        else:
            st.warning('Please go to the "Predict Price" tab and provide input features first to view top 5 Airbnb Listings in the seleted Neighborhood Group and Room Type')
    else:
        st.warning('Please go to the "Predict Price" tab and provide input features first to view top 5 Airbnb Listings in the seleted Neighborhood Group and Room Type')

    # user input for neighborhood and room type
    st.markdown('<h3 style="font-size: 1.5em;">You can also explore prices across various neighborhood groups and room types. Choose from the options below to view the list of prices</h3>', unsafe_allow_html=True)
    selected_neighborhood = st.selectbox('Select Neighborhood', data['neighbourhood_group'].unique(), index=None, placeholder='Select One')
    selected_room_type = st.selectbox('Select Room Type', data['room_type'].unique(), index=None, placeholder='Select One')

    # filtering data based on user selections
    filtered_listings = data[(data['neighbourhood_group'] == selected_neighborhood) & (data['room_type'] == selected_room_type)].head(5)

    if st.button('View'):
        if not filtered_listings.empty:
            # displaying heading with information based on user selections
            st.markdown(f'<h3 style="font-size: 1.5em;">Top 5 Airbnb Listings in {selected_neighborhood} - {selected_room_type}</h3>', unsafe_allow_html=True)

            # renaming the columns before displaying the table
            filtered_listings_display = filtered_listings[['neighbourhood_group', 'neighbourhood', 'price']].reset_index(drop=True)
            filtered_listings_display = filtered_listings_display.rename(columns={
                'neighbourhood_group': 'Neighbourhood Group',
                'neighbourhood': 'Neighbourhood',
                'price': 'Price'
            })

            # displaying the filtered table
            st.table(filtered_listings_display.style.format({'Price': "${:.2f}"}))
        else:
            st.warning(f"Choose atleast one option from each to show the table")


# Visualizations Page
elif selected_tab == "Visualizations":
    st.subheader('Visualizations')

  
    if st.button('Price Distribution based on Selected Neighborhood'):
        st.subheader('Price Distribution based on Selected Neighborhood')

        # getting the selected neighborhood group
        selected_neighborhood_group = st.session_state.input_features.get('neighbourhood_group')

        if selected_neighborhood_group:
            st.write("Explore the variety of Airbnb listing prices in your chosen neighborhood! This graph showcases the distribution, giving you a visual understanding of the different price ranges and how many listings fall within each.")
            # filtering data for the selected neighborhood group
            selected_neighborhood_data = data[data['neighbourhood_group'] == selected_neighborhood_group]

            # plotting the price distribution
            plt.figure(figsize=(10, 6))
            plt.hist(selected_neighborhood_data['price'], bins=30, color='blue', edgecolor='black')
            plt.xlabel('Price')
            plt.ylabel('Number of Listings')
            plt.title(f'Price Distribution in {selected_neighborhood_group}')
            st.pyplot(plt.gcf())
        else:
            st.warning('Please go to the "Predict Price" tab and select a neighborhood group first.')

   # visualizing top 10 hosts in the selected neighborhood group
    if st.button('Top 10 Hosts in Selected Neighborhood Group'):
        st.subheader('Top 10 Hosts in Selected Neighborhood Group')

        # getting the selected neighborhood group
        selected_neighborhood_group = st.session_state.input_features.get('neighbourhood_group')

        if selected_neighborhood_group:
            st.write("This visualization displays the top 10 hosts in your selected neighborhood group based on the number of listings they offer. Explore the bar chart to see which hosts stand out in terms of the quantity of available listings, providing valuable insights into the most prominent contributors in your chosen area.")
            # filtering data for the selected neighborhood group
            selected_neighborhood_data = data[data['neighbourhood_group'] == selected_neighborhood_group]

            # getting the top 10 hosts names and convert to uppercase
            top_10_hosts = selected_neighborhood_data['host_name'].value_counts().head(10).index.str.upper()

            # plotting a bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(top_10_hosts, selected_neighborhood_data['host_name'].value_counts().head(10), color='blue')
            ax.set_ylabel('Number of Listings')
            ax.set_xlabel('Host Names')
            ax.set_title(f'Top 10 Hosts in {selected_neighborhood_group}')
            st.pyplot(fig)
        else:
            st.warning('Please go to the "Predict Price" tab and select a neighborhood group first.')

    
    if st.button('Average Prices of Different Room Types in Selected Neighborhood Group'):
        st.subheader('Average Prices of Different Room Types in Selected Neighborhood Group')

        # getting the selected neighborhood group
        selected_neighborhood_group = st.session_state.input_features.get('neighbourhood_group')

        if selected_neighborhood_group:
            st.write("Explore average prices for different room types in your selected neighborhood, helping you make informed decisions when setting Airbnb listing prices.")
            # filtering data for the selected neighborhood group
            selected_neighborhood_data = data[data['neighbourhood_group'] == selected_neighborhood_group]

            # group by 'Room Type' and calculate the average price for each room type
            average_prices = selected_neighborhood_data.groupby('room_type')['price'].mean().reset_index()

            # plotting the average prices
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(average_prices['room_type'], average_prices['price'], color='blue')
            ax.set_xlabel('Room Type')
            ax.set_ylabel('Average Price')
            ax.set_title(f'Average Prices of Different Room Types in {selected_neighborhood_group}')
            st.pyplot(fig)
        else:
            st.warning('Please go to the "Predict Price" tab and select a neighborhood group first.')

    if st.button('Distribution of Room Types across NYC'):
        st.write("Explore the diversity of room types across NYC! This interactive map visualizes the distribution of room types, allowing you to identify neighborhoods and room types at a glance. Hover over points to view specific information about room types and neighborhoods.")
                
        st.write("Zoom In/Out: Use the scroll wheel to zoom in or out for a closer look.")
        st.write("Hover Over Points: Get details about room types and neighborhoods by hovering over points on the map. Explore the city's diverse offerings!")
        st.write("If you deselect the other two room types from the legend, the map will dynamically update to display only the locations associated with that specific room type which is selected. This feature allows you to focus on and explore one room type at a time for a more detailed view.")
        st.write('Click on Home icon to reset the view')
        # plotly scatter plot - map
        fig = px.scatter_mapbox(
            data,
            lat='latitude',
            lon='longitude',
            color='room_type',
            color_discrete_sequence=px.colors.qualitative.Set1,
            title='Distribution of room types across NYC',
            mapbox_style="carto-positron",
            zoom=10,
            hover_name='neighbourhood_group',  # adding neighborhood information to the hover tooltip
            hover_data={'latitude': False, 'longitude': False, 'room_type': True, 'neighbourhood': True}  # hover data customization
        )

        fig.update_layout(
            mapbox_style="carto-positron",
            mapbox_zoom=10,
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )

        # rendering the plotly figure using st.plotly_chart
        st.plotly_chart(fig)
    

    if st.button('Demand for Room Types According to Prices'):
        st.write("Explore the demand for different room types based on prices!")
        st.write(" This bar chart visualizes the average prices and count of each room type, "
                 "providing valuable insights for your decision-making process. Hover over the bars to view the count, "
                 "helping you understand the popularity of each room type in relation to its average price.")
        room_type_stats = data.groupby('room_type').agg({'price': ['mean', 'count']}).reset_index()
        room_type_stats.columns = ['room_type', 'average_price', 'count']

        fig = px.bar(
            room_type_stats,
            x='room_type',
            y='average_price',
            title='Demand for Room Types According to Prices',
            labels={'average_price': 'Average Price', 'room_type': 'Room Type'},
            color='room_type',
            color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data={'count': True}  # showing the count of listings when hovering over the bars
        )

        # layout customization
        fig.update_layout(
            xaxis_title='Room Type',
            yaxis_title='Average Price',
            barmode='group'
        )

        # rendering the plotly figure using st.plotly_chart
        st.plotly_chart(fig)