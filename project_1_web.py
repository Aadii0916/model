import streamlit as st
import numpy as np
import pandas as pd
import joblib
model = joblib.load("sonar_model.pkl")
try:
    # Load the model that we trained and saved earlier.
    model = joblib.load('sonar_model.pkl')
except FileNotFoundError:
    # If the file isn't found, show an error on the webpage.
    st.error("Model file ('sonar_model.pkl') not found. Please run the model_training.py script first to create it.")
    # Stop the app from running further if the model can't be loaded.
    st.stop()


# --- 2. CREATE THE WEB APP INTERFACE ---

# Set the title that appears in the browser tab
st.set_page_config(page_title="Sonar Predictor")

# Display the main title on the webpage
st.title('Sonar Rock vs. Mine Prediction')

# Display some instructions for the user
st.write("Enter the 60 sonar signal values below, separated by commas.")

# Create a text input box for the user to enter data
input_data_str = st.text_input('Enter 60 comma-separated sonar values here:')


# --- 3. MAKE PREDICTIONS ---

# Create a button. The code inside this 'if' statement will only run when the button is clicked.
if st.button('Make Prediction'):
    # First, check if the user has actually entered something in the text box.
    if input_data_str:
        try:
            # Step A: Convert the user's input string into a list of numbers.
            # The .split(',') method breaks the string into a list wherever there's a comma.
            # float(val.strip()) converts each item into a number and removes any extra spaces.
            input_data_list = [float(val.strip()) for val in input_data_str.split(',')]
            
            # Step B: Check if the user provided exactly 60 values.
            if len(input_data_list) == 60:
                # Step C: Convert the Python list into a NumPy array, which is required by the model.
                input_data_np = np.asarray(input_data_list)
                
                # Step D: Reshape the array. The model expects a 2D array, so we reshape our 1D array.
                input_data_reshaped = input_data_np.reshape(1, -1)
                
                # Step E: Use the loaded model to make a prediction on the user's data.
                prediction = model.predict(input_data_reshaped)
                
                # Step F: Display the prediction result.
                st.subheader('Prediction Result:')
                if prediction[0] == 'R':
                    st.success('The object is a **Rock**')
                else:
                    st.warning('The object is a **Mine**')
            else:
                # If the user didn't enter 60 values, show an error.
                st.error(f"Error: You entered {len(input_data_list)} values. Please enter exactly 60 values.")
        
        except ValueError:
            # If the input contains non-numeric characters (like letters), show an error.
            st.error("Error: Please make sure you enter only valid numbers separated by commas.")
        
    else:
        # If the button is clicked but the input box is empty, show a warning.
        st.warning("Please enter the sonar values to get a prediction.")










# def load_model():
#     """Loads the pre-trained logistic regression model from a file."""
#     try:
#         model = joblib.load('sonar_model.pkl')
#         return model
#     except FileNotFoundError:
#         st.error("Model file not found. Please run the model_training.py script first to create it.")
#         return None

# # Load the model
# model = load_model()

# # Set up the Streamlit page
# st.set_page_config(page_title="Sonar Rock vs. Mine Predictor", layout="wide")

# # App title
# st.title('Sonar Rock vs. Mine Prediction')
# st.write("""
# Enter the 60 sonar signal values below to predict whether the object is a Rock (R) or a Mine (M).
# Please provide the values as a comma-separated list.
# """)

# # Input from user
# input_data_str = st.text_input('Enter 60 comma-separated sonar values:')

# # Prediction button
# if st.button('Make Prediction'):
#     if model is not None and input_data_str:
#         try:
#             # 1. Parse the input string into a list of floats
#             input_data_list = [float(val.strip()) for val in input_data_str.split(',')]
            
#             # 2. Check if the user entered exactly 60 values
#             if len(input_data_list) == 60:
#                 # 3. Convert the list to a numpy array
#                 input_data_np = np.asarray(input_data_list)
                
#                 # 4. Reshape the array as we are predicting for one instance
#                 input_data_reshaped = input_data_np.reshape(1, -1)
                
#                 # 5. Make the prediction
#                 prediction = model.predict(input_data_reshaped)
                
#                 # 6. Display the result
#                 st.subheader('Prediction Result')
#                 if prediction[0] == 'R':
#                     st.success('The object is a **Rock**')
#                 else:
#                     st.warning('The object is a **Mine**')
#             else:
#                 st.error(f"Error: You entered {len(input_data_list)} values. Please enter exactly 60 values.")
#         except ValueError:
#             st.error("Error: Please make sure you enter valid numbers separated by commas.")
#         except Exception as e:
#             st.error(f"An unexpected error occurred: {e}")
#     elif not input_data_str:
#         st.warning("Please enter the sonar values to get a prediction.")
