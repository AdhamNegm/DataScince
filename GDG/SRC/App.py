import streamlit as st
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

#load preprocessed datta, encoders, scaler, model
def load_resources():
#load encoders and scaler 
    le_title = joblib.load('D:\Projects\GDG\WorkSpace\le_title.pkl')
    le_wishlist = joblib.load('D:\Projects\GDG\WorkSpace\le_wishlist.pkl')
    scaler = joblib.load('D:\Projects\GDG\WorkSpace\scaler.pkl')

#load model
    model = joblib.load('WorkSpace/rf_model.pkl')

    return le_title, le_wishlist, scaler, model

def main():
    st.title('Udemy Course Rating Prediction')

#load resources
le_title, le_wishlist, scaler, model = load_resources()

#Take user Input
input_data = {}

#course title selection
available_titles = list('le_title.classes')
search_query = st.text_input('search for a course title')
filtered_titles = [title for title in available_titles if search_query.lower() in title.lower()]
selected_title = st.selectbox(
    'select course title',
    filtered_titles if search_query else available_titles,
)

if selected_title in available_titles:
    input_data['title'] = selected_title

else:
    st.error("invalid course title selected.")





input_data['num_subscribers'] = st.number_input('number of subscribers', min_value=0, value=100)
input_data['num_reviews'] = st.number_input('number of reviews', min_value=0, value=10)
input_data['num_published_practice_tests'] = st.number_input('number of practice tests', min_value=0, value=10)
input_data['price_detail__amount'] = st.number_input('Course Price', min_value=0.0, value=10.0, step=0.1)
input_data['is_wishlisted'] = st.selectbox('is Wishlisted',[False, True])

#create a dataframe with user input
input_df = pd.DataFrame([input_data])

#Encode and scale the input data
input_df['title'] = le_title.transform(input_df['title'][0])
input_df['is_wishlisted'] = int(input_df['is_wishlisted'][0])

numerical_cols = ['num_subscribers', 'num_reviews', 'price_detail__amount']
input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

# Prediction
if st.button('Predict Rating'):
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(input_df)
    st.success(f'predict course rating: {prediction[0]:.2f}')