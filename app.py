# Importing necessary libraries
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st

# Reading the data from a CSV file (assuming 'framingham.csv' contains heart disease-related data)
Heart_Data = pd.read_csv("framingham.csv")

# Dropping rows with missing values
Heart_Data = Heart_Data.dropna()

# Splitting the data into features (X) and target (Y) where 'TenYearCHD' is the target variable
X = Heart_Data.drop(columns="TenYearCHD")
Y = Heart_Data["TenYearCHD"]

# Splitting the data into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Handling the class imbalance by oversampling the minority class
over_sample = RandomOverSampler(sampling_strategy="minority")
X_over, Y_over = over_sample.fit_resample(X_train, Y_train)

# Initializing the RandomForestClassifier model
model = RandomForestClassifier()

# Training the model using the oversampled data
model.fit(X_over, Y_over)

# Making predictions on the test data
preds = model.predict(X_test)

# Calculating the accuracy of the model
Accur = accuracy_score(Y_test, preds)
print(f"The Accuracy of The Model Is {Accur}")

# Streamlit interface for the web app
st.header("Heart Disease Prediction")

# Dividing the layout into three columns for better organization of the input fields
col1, col2, col3 = st.columns(3)

# Input fields for user data (gender, age, education, smoking status, etc.)
gender = col1.selectbox("Enter your gender", ["Male", "Female"])
Age = col2.number_input("Enter Your Age")
education = col3.selectbox("Highest academic qualification", ["High school diploma", "Undergraduate degree", "Postgraduate degree", "PhD"])
isSmoker = col1.selectbox("Are you currently a smoker?", ["Yes", "No"])
yearsSmoking = col2.number_input("Number of daily cigarettes")
BPMeds = col3.selectbox("Are you currently on BP medication?", ["Yes", "No"])
stroke = col1.selectbox("Have you ever experienced a stroke?", ["Yes", "No"])
hyp = col2.selectbox("Do you have hypertension?", ["Yes", "No"])
diabetes = col3.selectbox("Do you have diabetes?", ["Yes", "No"])
chol = col1.number_input("Enter your cholesterol level")
sys_bp = col2.number_input("Enter your systolic blood pressure")
dia_bp = col3.number_input("Enter your diastolic blood pressure")
bmi = col1.number_input("Enter your BMI")
heart_rate = col2.number_input("Enter your resting heart rate")
glucose = col3.number_input("Enter your glucose level")

# Create a dataframe from user inputs to be used for prediction
df_pred = pd.DataFrame([[gender, Age, education, isSmoker, yearsSmoking, BPMeds, stroke, hyp, diabetes, chol, sys_bp, dia_bp, bmi, heart_rate, glucose]],
                       columns=['gender', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'])

# Converting categorical features to numeric values (gender, smoking status, BP meds, stroke, hypertension, diabetes)
df_pred['gender'] = df_pred['gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Rename "gender" column to "male" to match the column used during training
df_pred.rename(columns={'gender': 'male'}, inplace=True)

df_pred['currentSmoker'] = df_pred['currentSmoker'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['prevalentHyp'] = df_pred['prevalentHyp'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['prevalentStroke'] = df_pred['prevalentStroke'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['diabetes'] = df_pred['diabetes'].apply(lambda x: 1 if x == 'Yes' else 0)
df_pred['BPMeds'] = df_pred['BPMeds'].apply(lambda x: 1 if x == 'Yes' else 0)

# Function to map education levels to numeric values
def transform(data):
    if data == 'High school diploma':
        result = 0
    elif data == 'Undergraduate degree':
        result = 1
    elif data == 'Postgraduate degree':
        result = 2
    return result

# Apply the transformation to the education column
df_pred['education'] = df_pred['education'].apply(transform)

# Making the prediction based on user input
prediction = model.predict(df_pred)

# Displaying the prediction result when the 'Predict' button is clicked
if st.button('Predict'):
    if prediction[0] == 0:
        st.write('<p class="big-font">You likely will not develop heart disease in 10 years.</p>', unsafe_allow_html=True)
    else:
        st.write('<p class="big-font">You are likely to develop heart disease in 10 years.</p>', unsafe_allow_html=True)
