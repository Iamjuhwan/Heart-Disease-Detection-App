import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle
import sklearn
import joblib

st.image('picture.jpg')
st.title(':blue[Heart disease Prediction App]')
st.write("""-- This app predicts a Patient has a heart disease or not --

""")
st.write(':point_left: (click arrow sign for hide and unhide form) :green[Please Fillup the input field of left side for Prediction.] :sunglasses:')
st.download_button('Download Sample file link for check', 'https://github.com/Iamjuhwan/Heart-Disease-Detection-App/blob/main/heart_disease_dataset.csv')
st.sidebar.header('Please Input Features Value')

# Collects user input features into dataframe

def user_input_features():
    age = st.sidebar.number_input('Age of persons: ')
    sex = st.sidebar.selectbox('Gender of persons 0=Female, 1=Male: ',(0,1))
    cp = st.sidebar.selectbox('Chest pain type (4 values)',(0,1,2,3))
    trtbps = st.sidebar.number_input('Resting blood pressure: ')
    chol = st.sidebar.number_input('Serum cholestrol in mg/dl: ')
    fbs =  st.sidebar.selectbox('Fasting blood sugar > 120 mg/dl:',( 0,1))
    restecg = st.sidebar.selectbox('Resting electrocardio results:', ( 0,1,2))
    thalachh = st.sidebar.number_input('Maximum heart rate achieved thalach: ')
    exng = st.sidebar.selectbox('Exercise induced angina: ',( 0,1))
    oldpeak = st.sidebar.number_input(' ST depression induced by exercise relative to rest (oldpeak): ')
    slp = st.sidebar.selectbox('The slope of the peak exercise ST segment (slp): ', ( 0,1,2))
    caa = st.sidebar.selectbox('Number of major vessels(0-3) colored by flourosopy (caa):',(0,1,2,3,4))
    thall = st.sidebar.selectbox(' Thall 0=normal, 1=fixed defect, 2 = reversable defect',(0,1,2,3))


    data = {'age':age, 'sex':sex, 'cp':cp, 'trtbps':trtbps, 'chol':chol, 'fbs':fbs, 'restecg':restecg, 'thalachh':thalachh,
       'exng':exng, 'oldpeak':oldpeak, 'slp':slp, 'caa':caa, 'thall':thall
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

st.write(input_df)

def predict(data):
    clf = joblib.load("model_LogR.sav")
    return clf.predict(data)
