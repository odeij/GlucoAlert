import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Creating a function for prediction
def diabetes_predictions(input_data):
    # Convert the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make a prediction
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

def main():
    # Title for the app
    st.title('GlucoAlert Web App')

    # Get input data from the user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function')
    Age = st.text_input('Age')

    # Initialize diagnosis message
    diagnosis = ''

    # Create a button for prediction
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to numeric values
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age),
            ]
            diagnosis = diabetes_predictions(input_data)
        except ValueError:
            diagnosis = 'Please enter valid numeric values for all fields.'

    st.success(diagnosis)
    st.info("Always remember to consult your doctor.")

if __name__ == '__main__':
    main()
