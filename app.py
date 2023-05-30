# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import plotly.express as px
import pandas as pd
import joblib
import streamlit as st
from streamlit_option_menu import option_menu
import sys
import matplotlib.pyplot as plt
import plotly.graph_objs as go

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')



# load models
purchase_day_ml = joblib.load('models/Customer_next_purchase_day_model.joblib_1')
ltv_model = joblib.load('models/Customer_lifetime_value_model.joblib_1')


# read data
featured_data = pd.read_csv('C:/Users/user/Data_Driven/artifacts/featured_data.csv')



st.title('CusAnalytics')
st.write('A Machine Learning Web App for Predicting Customer Lifetime Value and Next Purchase Trends. Treat your customers in a way they deserve before they expect that and act before something bad happens.')

# creating sidebar

with st.sidebar :
    selector = option_menu(('Future-Proof Your Business'),
                           
                           ['Exploratory Analysis',
                            'Segmentation',
                            'Predictors',
                               ], icons=["📈",'📊','🔍'])
    
    
#analysis
customer = (pd.concat([pd.Series('ALL'),featured_data['CustomerID']]))
if selector == ('Exploratory Analysis'):
    st.title('Explore Your Data')
    select = st.selectbox('Select or search for a customerID from the the database',options=customer)
    if select == 'ALL' :
        st.write('see some important metrics and chart about the whole data connected')
        col1,col2,col3 = st.columns(3)
        tab1,tab2,tab3 = st.tabs(['Recency','Frequency','Revenue'])
        with col1 :
            st.metric('Number of unique customers',len(featured_data['CustomerID']))
        with col2:
            st.metric('Number of purchases made',featured_data['Frequency'].sum())
        with col3 :
                    st.metric('Total Revenue',round(featured_data['TotalRevenue'].sum()))
        with tab1:
            fig = px.scatter(featured_data, x="Recency", y="Frequency", color="RecencyCluster",
                             hover_data=['Recency'])
            st.plotly_chart(fig)
            fig = px.scatter(featured_data, x="Recency", y="TotalRevenue", color="RecencyCluster",
                             hover_data=['Recency'])
            st.plotly_chart(fig)
        with tab2:
                fig = px.scatter(featured_data, x="Frequency", y="Recency", color="FrequencyCluster",
                                 hover_data=['Frequency'])
                st.plotly_chart(fig)
                fig = px.scatter(featured_data, x="Frequency", y="TotalRevenue", color="FrequencyCluster",
                                 hover_data=['Frequency'])
                st.plotly_chart(fig)
        with tab3:
                    fig = px.scatter(featured_data, x="TotalRevenue", y="Recency", color="TotalRevenueCluster",
                                     hover_data=['TotalRevenue'])
                    st.plotly_chart(fig)
                    fig = px.scatter(featured_data, x="TotalRevenue", y="Frequency", color="TotalRevenueCluster",
                                     hover_data=['TotalRevenue'])
                    st.plotly_chart(fig)
    else :
        col1,col2,col3 = st.columns(3)
        cus_info = featured_data[featured_data['CustomerID']==select]
        with col1 :
            st.metric('Number of unique customers',len(cus_info['CustomerID']))
        with col2:
            st.metric('Number of purchases made',cus_info['Frequency'])
        with col3 :
            st.metric('Total Revenue',round(cus_info['TotalRevenue']))

if selector == ('Segmentation'):   
    st.title('Customer Segementation')
    low_value = featured_data[featured_data['Segment_low_value']==1].reset_index(drop='index')
    mid_value = featured_data[featured_data['Segment_Mid-Value']==1].reset_index(drop='index')
    high_value = featured_data[featured_data['Segment_High-Value']==1].reset_index(drop='index')
    
    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric('Numbor low value customers',len(low_value))
    with col2:
        st.metric('Number of mid value customers',len(mid_value))
    with col3:
        st.metric('Number of high value customers',len(high_value))
        
    labels = ['low value','Mid value','High value']
    values = [len(low_value),len(mid_value),len(high_value)]
    # Create a bar chart using Matplotlib
    # Create the Plotly pie chart object
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    
    # Add a title to the chart
    fig.update_layout(title='Customer Segmentation')
    
    # Use Streamlit to display the chart
    st.plotly_chart(fig, use_container_width=True)
    segment = ['ALL','High value','Mid value','Low value']
    select = st.selectbox('Select Customer Segement',options=segment)
    if select == 'ALL':
        st.dataframe(featured_data)
    elif select == 'High value':
        st.dataframe(high_value)
    elif select == 'Mid value':
        st.dataframe(mid_value)
    else:
        st.dataframe(low_value)
        
if selector == 'Predictors':
    estimators = ['30_day_purchase','60_day_purchase','90_day_purchase','ltv_predictor']
    select = st.selectbox('choose the estimator you want to use',estimators)
    if select in ['30_day_purchase','60_day_purchase','90_day_purchase']:
        preds = purchase_day_ml.predict(featured_data)
        proba = purchase_day_ml.predict_proba(featured_data)
        featured_data['predictions'] = preds
        # Find the index of the highest probability in each row of the proba array
        max_proba_idx = np.argmax(proba, axis=1)

        # Select the highest probability from the proba array using the max_proba_idx
        max_proba = proba[np.arange(len(proba)), max_proba_idx]

        # Assign the max_proba values to a new column in the featured_data data frame
        featured_data['preds_proba'] = max_proba
        if select == '30_day_purchase':
            df = featured_data[featured_data['predictions'] == 0].reset_index(drop='index')
            st.dataframe(df)
            csv = convert_df(df)
            st.download_button('Download Prediction',data=csv,file_name='customer purchase day.csv')
        if select == '60_day_purchase':
            df = featured_data[featured_data['predictions'] == 1].reset_index(drop='index')
            st.dataframe(df)
            csv = convert_df(df)
            st.download_button('Download Prediction',data=csv,file_name='customer purchase day.csv')
        if select == '90_day_purchase':
            df = featured_data[featured_data['predictions'] == 2].reset_index(drop='index')
            st.dataframe(df.drop_duplicates())
            csv = convert_df(df)
            st.download_button('Download Prediction',data=csv,file_name='customer purchase day.csv')
    if select == 'ltv_predictor':
        preds = ltv_model.predict(featured_data)
        proba = ltv_model.predict_proba(featured_data)
        df = featured_data.copy()
        df['predictions'] = preds
        # Find the index of the highest probability in each row of the proba array
        max_proba_idx = np.argmax(proba, axis=1)
        # Select the highest probability from the proba array using the max_proba_idx
        max_proba = proba[np.arange(len(proba)), max_proba_idx]
        # Assign the max_proba values to a new column in the featured_data data frame
        df['preds_proba'] = max_proba
        st.dataframe(df)
        csv = convert_df(df)
        st.download_button('Download Prediction',data=csv,file_name='Life_time_value.csv')
        
        
            
                  