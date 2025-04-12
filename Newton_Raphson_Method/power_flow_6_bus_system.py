"""
Created on Wed Apr  9 13:51:19 2025

@author: ederrenato
"""
import pandas as pd
import numpy as np

line_data_path = "./line_data_of_ieee_6_bus_system.csv"
bus_data_path = "./bus_data_of_ieee_6_bus_system.csv"


line_data_df = pd.read_csv(line_data_path)
bus_data_df = pd.read_csv(bus_data_path)

line_data_array = np.array(line_data_df)
bus_data_array = np.array(bus_data_df)

class ImpedanceMatrix:
    def __init__(self, imp_array):
        self.imp_array = imp_array


