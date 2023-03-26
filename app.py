import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# Heart disease Prediction App

This app predicts If a patient has a heart disease or not
""")

st.header('User Input Features')



# Collects user input features into dataframe

def user_input_features():
    age = st.number_input('Enter your age: ')
    sex_mapping = {0: 'M', 1: 'F'}
    sex = st.selectbox('Sex', options=list(sex_mapping.keys()), format_func=lambda x: sex_mapping[x])
    cp = st.selectbox('Chest pain type',(0,1,2,3))
    tres = st.number_input('Resting blood pressure: ')
    chol = st.number_input('Serum cholestoral in mg/dl: ')
    fbs = st.selectbox('Fasting blood sugar',(0,1))
    res = st.number_input('Resting electrocardiographic results: ')
    tha = st.number_input('Maximum heart rate achieved: ')
    exa = st.selectbox('Exercise induced angina: ',(0,1))
    old = st.number_input('oldpeak ')
    slope = st.number_input('he slope of the peak exercise ST segmen: ')
    ca = st.selectbox('number of major vessels',(0,1,2,3))
    thal = st.selectbox('thal',(0,1,2))

    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

ok = st.button("Predict")

if ok:

    # Combines user input features with entire dataset
    # This will be useful for the encoding phase
    heart_dataset = pd.read_csv('heart.csv')
    heart_dataset = heart_dataset.drop(columns=['target'])

    df = pd.concat([input_df,heart_dataset],axis=0)

    # Encoding of ordinal features
    # https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
    df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

    df = df[:1] # Selects only the first row (the user input data)

    st.write(input_df)
    # Reads in saved classification model
    load_clf = pickle.load(open('Random_forest_model.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)


    st.subheader('Prediction')
    st.write(prediction)
    if(prediction==1):
        st.write("""You are having a Heart Disease""")
    else:
        st.write("""You are not having a Heart Disease""")

    st.subheader('Prediction Probability')
    st.write(prediction_proba)
