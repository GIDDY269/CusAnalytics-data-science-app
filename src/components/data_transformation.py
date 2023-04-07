import sys
sys.path.append('C:/Users/user/Data_Driven')
import pandas as pd
import numpy as np
import os 
from dataclasses import dataclass
from sklearn.cluster import KMeans


@dataclass 
class DataTransformationconfig:
    file_path = 'C:/Users/user/Data_Driven/artifacts'
    data_transform_object_path = os.path.join(file_path,'featured_data.csv')
    
class DataTransform:
    def __init__(self):
        self.data_transfrom_object = DataTransformationconfig()
        
    def cluster(self,column,df,n=4):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(df[[column]])
        df[column + 'Cluster'] = kmeans.predict(df[[column]])
        return df
    
    def order_cluster(self,df, target_field_name, cluster_field_name, ascending):
        """
        INPUT:
            - df                  - pandas DataFrame
            - target_field_name   - str - A column in the pandas DataFrame df
            - cluster_field_name  - str - Expected to be a column in the pandas DataFrame df
            - ascending           - Boolean
            
        OUTPUT:
            - df_final            - pandas DataFrame with target_field_name and cluster_field_name as columns
        
        """
        # Add the string "new_" to cluster_field_name
        new_cluster_field_name = "new_" + cluster_field_name
        
        # Create a new dataframe by grouping the input dataframe by cluster_field_name and extract target_field_name 
        # and find the mean
        df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
        
        # Sort the new dataframe df_new, by target_field_name in descending order
        df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
        
        # Create a new column in df_new with column name index and assign it values to df_new.index
        df_new["index"] = df_new.index
        
        # Create a new dataframe by merging input dataframe df and part of the columns of df_new based on 
        # cluster_field_name
        df_final = pd.merge(df, df_new[[cluster_field_name, "index"]], on=cluster_field_name)
        
        # Update the dataframe df_final by deleting the column cluster_field_name
        df_final = df_final.drop([cluster_field_name], axis=1)
        
        # Rename the column index to cluster_field_name
        df_final = df_final.rename(columns={"index": cluster_field_name})
        
        return df_final
        
    def initiate_data_transfrom(self,data):
        dataframe = pd.read_csv(data)
        dataframe['InvoiceDate'] = pd.to_datetime(dataframe['InvoiceDate'])
        dataframe['Revenue'] = dataframe['Price'] * dataframe['Quantity']

        # Creating cohort analysis
        cus_past_df = dataframe[(dataframe['InvoiceDate'] >= pd.Timestamp(2009,12,1)) & (dataframe['InvoiceDate'] < 
                                                                                        pd.Timestamp(2011,9,1))].reset_index(drop=True)

        cus_next_quarter = dataframe[(dataframe['InvoiceDate'] >= pd.Timestamp(2011,9,1)) & (dataframe['InvoiceDate'] < 
                                                                                        pd.Timestamp(2011,12,9))].reset_index(drop=True)
        
        # get the distinst customers in cus_past_df
        cus_df = pd.DataFrame(cus_past_df['Customer ID'].unique())
        cus_df.columns  = ['Customer ID']
        
        
        # Create a dataframe with CustomerID and customers first purchase 
        # date in cus_next_quarter
        cus_1st_purchase_in_next_quarter = cus_next_quarter.groupby('Customer ID')['InvoiceDate'].min().reset_index()
        cus_1st_purchase_in_next_quarter.columns = ['Customer ID','FirstPurchaseDate']
        
        
        cus_last_purchase_past_df = cus_past_df.groupby('Customer ID')['InvoiceDate'].max().reset_index()
        cus_last_purchase_past_df.columns = ['Customer ID','LastPurchaseDate']
        
        # Merge two dataframes cus_last_purchase_past_df and cus_1st_purchase_in_next_quarter
        cus_purchase_dates = pd.merge(cus_last_purchase_past_df,cus_1st_purchase_in_next_quarter,on='Customer ID',how = 'left')
       #  time difference in days between customer's last purchase in the dataframe cus_last_purchase_past_df and the first purchase in the dataframe cus_1st_purchase_in_next_quarter.
        cus_purchase_dates['NextPurchaseDay'] = (cus_purchase_dates['FirstPurchaseDate'] - cus_purchase_dates['LastPurchaseDate']).dt.days
        
        # merge with cus_df
        cus_df = pd.merge(cus_df,cus_purchase_dates[['Customer ID','NextPurchaseDay']],on='Customer ID',how='left')
        cus_df.fillna(9999,inplace=True) #fill missing values
        
        # Recency
        cus_max_purchase = cus_past_df.groupby('Customer ID')['InvoiceDate'].max().reset_index()
        cus_max_purchase.columns = ['Customer ID','LastPurchaseDate']
        cus_max_purchase['Recency'] = (cus_max_purchase['LastPurchaseDate'].max()-cus_max_purchase['LastPurchaseDate']).dt.days
        cus_df = pd.merge(cus_df,cus_max_purchase[['Customer ID','Recency']],on='Customer ID')
        
        # recencycluster
        cus_df = self.cluster('Recency', cus_df)
        cus_df= self.order_cluster(cus_df, 'Recency', 'RecencyCluster', False)
        
        # Frequency
        #get order counts for each user and create a dataframe with it
        cus_frequency = cus_past_df.groupby('Customer ID')['InvoiceDate'].count().reset_index()
        cus_frequency.columns = ['Customer ID','Frequency']
        cus_df = pd.merge(cus_df,cus_frequency,on = 'Customer ID')
        cus_df = self.cluster('Frequency', cus_df)
        cus_df = self.order_cluster(cus_df, 'Frequency', 'FrequencyCluster', True)

        # Revenue
        cus_revenue = cus_past_df.groupby('Customer ID')['Revenue'].sum().reset_index()
        cus_df = pd.merge(cus_df,cus_revenue,on='Customer ID')
        cus_df.rename(columns={'Revenue':'TotalRevenue'},inplace=True)

        cus_revenue_mean = cus_past_df.groupby('Customer ID')['Revenue'].mean().reset_index()
        cus_df = pd.merge(cus_df,cus_revenue_mean,on='Customer ID')
        cus_df.rename(columns={'Revenue':'MeanRevenue'},inplace =True)
        cus_df = self.cluster('TotalRevenue', cus_df)
        cus_df = self.order_cluster(cus_df, 'TotalRevenue', 'TotalRevenueCluster', True)

        # Overall score
        cus_df['OverallScore'] = cus_df['RecencyCluster'] + cus_df['FrequencyCluster'] + cus_df['TotalRevenueCluster']
        cus_df['Segment'] = 'low_value'
        cus_df.loc[cus_df['OverallScore'] > 2, 'Segment'] = 'Mid-Value'
        cus_df.loc[cus_df['OverallScore'] > 4, 'Segment'] = 'High-Value'  
        
        #create cus_data as a copy of cus_df before applying get_dummies
        cus_data = cus_df.copy()
        cus_data = pd.get_dummies(cus_data)
        cus_data.drop('NextPurchaseDay',axis = 1,inplace = True)
        
        cus_data.to_csv(self.data_transfrom_object.data_transform_object_path,header=True,index=False)
        
        return(cus_data,self.data_transfrom_object)
