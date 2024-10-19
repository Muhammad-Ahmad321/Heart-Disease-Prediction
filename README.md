Heart Disease Prediction App
This is a web-based application built using Streamlit and RandomForestClassifier from scikit-learn to predict the likelihood of developing heart disease within 10 years based on various health and lifestyle inputs. The application utilizes the Framingham Heart Study dataset to train the machine learning model.

Features
Input features such as age, gender, smoking status, blood pressure, cholesterol levels, and more.
Predict whether the user is at risk of developing heart disease in the next 10 years.
Simple and interactive interface built using Streamlit.
The model handles class imbalance using RandomOverSampler from imblearn.
How to Run
1. Clone the Repository

git clone https://github.com/Muhammad-Ahmad321/Heart-Disease-Prediction.git
cd heart-disease-prediction-app
2. Install Dependencies
Make sure you have Python 3.7 or later installed. Install the required packages using pip:


pip install -r requirements.txt
3. Run the App
Run the Streamlit app using the following command:


streamlit run app.py
4. Use the App
The application will launch in your web browser. You can input your health and lifestyle information, and the app will predict your heart disease risk.

Dataset
The app uses the Framingham Heart Study dataset (framingham.csv) which is included in the project. The dataset includes the following features:

age: Age of the person.
male: Gender (1 for male, 0 for female).
currentSmoker: Whether the person is a smoker (1 for yes, 0 for no).
cigsPerDay: Number of cigarettes smoked per day (for smokers).
BPMeds: Whether the person is on blood pressure medication.
prevalentStroke: Whether the person has experienced a stroke.
prevalentHyp: Whether the person has hypertension.
diabetes: Whether the person has diabetes.
totChol: Total cholesterol level.
sysBP: Systolic blood pressure.
diaBP: Diastolic blood pressure.
BMI: Body Mass Index.
heartRate: Resting heart rate.
glucose: Glucose level.
TenYearCHD: Target variable indicating whether the person will develop heart disease in 10 years.
Requirements
Python 3.7 or later
Libraries mentioned in the requirements.txt file.
File Structure
bash
Copy code
├── app.py               # Main application file
├── framingham.csv        # Dataset used for training
├── requirements.txt      # Python dependencies
├── README.md             # Documentation file
Model Training
The app uses a RandomForestClassifier for prediction. Before making predictions, the dataset is split into features (X) and the target (Y), followed by oversampling of the minority class to handle class imbalance using RandomOverSampler.

Acknowledgements
This project is based on the Framingham Heart Study dataset, which has contributed immensely to cardiovascular research.

Contact
For any questions or issues, feel free to open an issue in the repository or contact the developer at ahmed.johar1133@example.com.

