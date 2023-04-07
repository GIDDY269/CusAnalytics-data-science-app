import sys
sys.path.append('C:/Users/user/Data_Driven')
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationconfig
from src.components.data_transformation import DataTransform


@dataclass
class datainjectionconfig:
    file_path = 'C:/Users/user/Data_Driven/artifacts'
    data_file_path = os.path.join(file_path,'data.csv')
    
class DataInjection:
    def __init__(self):
        self.data_injection_path = datainjectionconfig()
        
    def InitiateDataInjection(self):
        # read data
        df = pd.read_csv('C:/Users/user/Data_Driven/online_retail_II.csv')
        
        
        os.makedirs(self.data_injection_path.file_path, exist_ok=True)
        os.makedirs(os.path.dirname(self.data_injection_path.data_file_path), exist_ok=True)
        
        df.to_csv(self.data_injection_path.data_file_path,header=True,index=False)
        
        return(self.data_injection_path.data_file_path)
    
    
if __name__=='__main__':
    obj = DataInjection()
    data = obj.InitiateDataInjection()
    
    transform = DataTransform()
    transform.initiate_data_transfrom(data)