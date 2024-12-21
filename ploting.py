import numpy as np
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# Function for prediction
def diabetes_predictions(input_data):
    # Changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Main function
def main():
    # Title
    st.title('GlucoAlert Web App')
    
    # Input data from the user
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose level')
    BloodPressure = st.text_input('Blood pressure value')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Pedigree Function')
    Age = st.text_input('Age')
    
    # Code for prediction
    diagnosis = ''
    
    # Create a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_predictions([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(diagnosis)
    
    # Visualization Section
    st.subheader("Data Visualization")
    
    # Example Dataset
    sample_data = {
        'Pregnancies': [int(Pregnancies)] if Pregnancies else [],
        'Glucose': [int(Glucose)] if Glucose else [],
        'BloodPressure': [int(BloodPressure)] if BloodPressure else [],
        'SkinThickness': [int(SkinThickness)] if SkinThickness else [],
        'Insulin': [int(Insulin)] if Insulin else [],
        'BMI': [float(BMI)] if BMI else [],
        'DiabetesPedigreeFunction': [float(DiabetesPedigreeFunction)] if DiabetesPedigreeFunction else [],
        'Age': [int(Age)] if Age else [],
    }
    df = pd.DataFrame(sample_data)
    
    if not df.empty:
        st.write("Input Data Summary:")
        st.dataframe(df)
        
        # Plot Interaction: Insulin vs Age
        if 'Insulin' in df.columns and 'Age' in df.columns:
            st.write("Interactive Scatter Plot: Insulin vs Age")
            scatter_fig = px.scatter(
                df,
                x='Age',
                y='Insulin',
                title='Insulin Levels vs Age',
                labels={'Age': 'Age', 'Insulin': 'Insulin Levels'},
                size='Insulin',
                color='Age',
                hover_data=['Insulin']
            )
            scatter_fig.update_layout(
                title=dict(x=0.5),
                template="plotly_white",
                font=dict(size=14)
            )
            st.plotly_chart(scatter_fig)

        # Plot Interaction: Glucose vs BMI
        if 'Glucose' in df.columns and 'BMI' in df.columns:
            st.write("Interactive Scatter Plot: Glucose vs BMI")
            scatter_fig = px.scatter(
                df,
                x='Glucose',
                y='BMI',
                title='Glucose Levels vs BMI',
                labels={'Glucose': 'Glucose Level', 'BMI': 'Body Mass Index'},
                size='BMI',
                color='Glucose',
                hover_data=['BMI', 'Glucose']
            )
            scatter_fig.update_layout(
                title=dict(x=0.5),
                template="plotly_white",
                font=dict(size=14)
            )
            st.plotly_chart(scatter_fig)

        # Plot Interaction: BloodPressure vs Age
        if 'BloodPressure' in df.columns and 'Age' in df.columns:
            st.write("Interactive Line Plot: Blood Pressure vs Age")
            line_fig = px.line(
                df,
                x='Age',
                y='BloodPressure',
                title='Blood Pressure vs Age',
                labels={'Age': 'Age', 'BloodPressure': 'Blood Pressure'},
                markers=True
            )
            line_fig.update_layout(
                title=dict(x=0.5),
                template="plotly_white",
                font=dict(size=14)
            )
            st.plotly_chart(line_fig)

if __name__ == '__main__':
    main()
