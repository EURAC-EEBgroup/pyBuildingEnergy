"""Main module."""


# #%%
# # import sys
# # print(sys.path)
# from pybuildingenergy.src.utils import __ISO52010__, __ISO52016__, bui_item
# from pybuildingenergy.src.building_stock import Building_archetype

# from pybuildingenergy.src.functions import Filter_list_by_indices, Scatter_with_regression,Simple_regeression
# from scipy.stats import linregress

# # GET ARCHETYPE
# inizialize = Building_archetype('single_fammily_house','before 1900',44.66345144066082, 10.323822015417987)
# inputs_archetype = inizialize.Get_archetype()

# inputs_user = {
#     'url_api': "http://127.0.0.1:8000/api/v1", 
#     'latitude':46.66345144066082,
#     'longitude':9.71636944229362,
#     'Eln':10, #
#     'a_use': 100, 
#     "slab_on_ground_area":100,#
#     'H_setpoint':22,      
#     'C_setpoint':24,
#     'Phi_H_nd_max':40000,            
#     'Phi_C_nd_max':-10000,
# }

# inputs_user = []

# new_inputs = inizialize.Set_own_values(inputs_archetype,inputs_user)
# #%%
# # SIMULATE ARCHETYPE
# hourly_sim = __ISO52016__(inputs_archetype).Temperature_and_Energy_needs_calculation() 






#%%
from src.utils import __ISO52010__, __ISO52016__, bui_item, MyClass
from data.building_archetype import Selected_bui_archetype
from src.graphs import __Graphs__

# GET OBJECT ARCHETYPE
inizialize_building = Selected_bui_archetype('single_fammily_house','before 1900',45.47450206750902, 9.166056263772619)
BUI = inizialize_building.Get_bui_archetype()
## inputs_user = {
#     'latitude':46.66345144066082,
#     'longitude':9.71636944229362,
#     'a_use': 100, 
#     "slab_on_ground":100,#
#     'H_setpoint':22,     
#     'C_setpoint':24,
#     'Phi_H_nd_max':40000,            
#     'Phi_C_nd_max':-10000,
#     'volume':400,
#     'A_eli': [0, 76.69, 53.3, 53.3, 69.5, 73.46, 1.74, 5.22]
# }
# BUI.update_values(inputs_user)
# BUI.inputs_validation()
# BUI.A_eli

#%%
hourly_sim = __ISO52016__().Temperature_and_Energy_needs_calculation(BUI) 
__Graphs__(hourly_sim,'heating_cooling').bui_analysis_page()
# hourly_sim.to_csv('test.csv')
# print(hourly_sim, sum(hourly_sim['Q_H'])/(1000*BUI.a_use))





#%%


# # Plot
# import plotly.graph_objects as go

# # Create trace for variable 1
# trace1 = go.Scatter(x=df1.index, y=df1.loc[:,'Q_H'], mode='lines', name='Variable 1')


# # Create layout
# layout = go.Layout(title='Time Series Chart',
#                    xaxis=dict(title='Timestamp'),
#                    yaxis=dict(title='Value'))

# # Create figure
# fig = go.Figure(data=[trace1], layout=layout)

# # Show the plot
# fig.show()
# #                                               SCATTER WITH REGRESSION
# ========================================================================================================



# #%%'
# # SIMPLE REGRESSION
# data = hourly_sim.loc[:,['Q_H', 'T_ext']]

# # removing 0 values
# df_1 = data.loc[data['Q_H']>0,:]
# # daily values 
# df_2 = data.resample('D').mean()
# df_2['HDD'] = Filter_list_by_indices(df_2['T_ext'].values.tolist()) 
# df_2 = df_2.loc[df_2['Q_H']>0,:]
# # monthly_values
# df_3 = df_2.resample('ME').mean() # for temperature
# df_3_HDD = df_2.resample('ME').sum() # for temperature
# df_3_final = df_3.loc[:,['Q_H', 'T_ext']]
# df_3_final['HDD'] = df_3_HDD.loc[:,'HDD']
# df_3_final = df_3_final.loc[df_3_final['HDD']>1, :]

# x_data = df_2['Q_H'].values.tolist()
# y_data = [df_2['HDD'].values.tolist()]
# slope, intercept, r_value, p_value, std_err = linregress(x_data, y_data)

# colorRegression = ['black']
# colorPoints = ['green']
# seriesName = "Heating Consumption"
# name_y_axes = "Energy kWh"

# Scatter_with_regression(x_data, y_data, colorPoints, colorRegression,seriesName,name_y_axes, graphName="regressionScatter.html")
# Simple_regeression(x_data, y_data, "HDD")

# # import pandas as pd
# df1 = pd.read_csv('data_test_1.csv',index_col=0, header=0)
# df2 = pd.read_csv('test_data_unique.csv',index_col=0, header=0)
# # Plot
# import plotly.graph_objects as go

# # Create trace for variable 1
# trace1 = go.Scatter(x=df1.index, y=df1['Q_H'], mode='lines', name='Variable 1')

# # Create trace for variable 2
# trace2 = go.Scatter(x=df1.index, y=df2['Q_H'], mode='lines', name='Variable 2')

# # Create layout
# layout = go.Layout(title='Time Series Chart',
#                    xaxis=dict(title='Timestamp'),
#                    yaxis=dict(title='Value'))

# # Create figure
# fig = go.Figure(data=[trace1], layout=layout)

# # Show the plot
# fig.show()
# %%
