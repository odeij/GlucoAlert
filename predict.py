# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:44:31 2024

@author: leb
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# creating a function for prediction

def diabetes_predictions(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    #giving a title 
    
    st.title('GlucoAlert web App')
    
    # getting the input data from the user
    
    
    Pregnancies = st.text_input('Number of pregnancies ')
    Glucose = st.text_input('Glucose level ')
    BloodPressure= st.text_input('Blood pressure value ')
    SkinThickness = st.text_input('Skin Thickness ')
    Insulin= st.text_input('Insulin level ')
    BMI= st.text_input('BMI ')
    DiabetesPedigreeFunction= st.text_input('PedigreeFunction ')
    Age = st.text_input('Age')
    
    
    #code for prediction
    
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Diabetes Test Result '):
        diagnosis = diabetes_predictions([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    st.info("Always Remember to consult your doctor. ")
    
    

    
    
if __name__ == '__main__':
    main()
        
    
    
