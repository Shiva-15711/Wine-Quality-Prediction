import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import pickle
from sklearn.preprocessing import MinMaxScaler
from Decision_tree_model import *

st.set_page_config(page_title='Wine quality Predictor')
st.header('Wine quality Predictor')

excel_file = 'Winequality_dataset_final.xlsx'
sheet_name = 'data'

data  = pd.read_excel(excel_file,
                    sheet_name = sheet_name,
                    usecols = 'A:K',
                    header = 0)


list_of_columns = data.columns
input_data=pd.DataFrame(columns=list_of_columns)
input_data.drop(['quality'], axis='columns', inplace=True)

input_data.at[0, 'fixed acidity'] = st.slider('Fixed Acidity:',
                                        min_value = 0,
                                        max_value = 20,
                                        )
input_data.at[0, 'volatile acidity'] = st.slider('Volatile Acidity:',
                                        min_value = 0,
                                        max_value = 10,
                                        )
input_data.at[0, 'citric acid'] = st.slider('Citric Acid :',
                                        min_value = 0,
                                        max_value = 10,
                                        )
input_data.at[0, 'residual sugar'] = st.slider('Residual Sugar :',
                                        min_value = 0,
                                        max_value = 10,
                                        )
input_data.at[0, 'chlorides'] = st.slider('Chlorides:',
                                        min_value = 0,
                                        max_value = 5,
                                        )
input_data.at[0, 'free sulfur dioxide'] = st.slider('Free Suplur Dioxide :',
                                        min_value = 0,
                                        max_value = 50,
                                        )
input_data.at[0, 'total sulfur dioxide'] = st.slider('Total Sulphur Dioxide :',
                                        min_value = 0,
                                        max_value = 500,
                                        )
input_data.at[0, 'density'] = st.slider('Density :',
                                        min_value = 0,
                                        max_value = 5,
                                        )
input_data.at[0, 'sulphates'] = st.slider('Sulphates :',
                                        min_value = 0,
                                        max_value = 5,
                                        )
input_data.at[0, 'alcohol'] = st.slider('Alcohol :',
                                        min_value = 0,
                                        max_value = 30,
                                        )

if st.button("Predict Wine Quality"):
    y_pred =  clf.predict(input_data)
    quality = y_pred*(quality_max-quality_min)+quality_min
    if quality == 1:
        st.text('The wine quality is good.')
    else:
        st.text('The quality of wine is bad.')