import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load and preprocess the dataset
data = pd.read_csv('healthcare_dataset.csv')

# Map "Test Results" to numerical categories
data['Test Results'] = data['Test Results'].map({'Normal': 1, 'Inconclusive': 0, 'Abnormal': 2})

# Keep relevant columns for prediction
relevant_columns = ['Room Number', 'Date of Admission', 'Discharge Date', 'Age', 'Gender', 'Blood Type', 'Medical Condition', 'Test Results']
data_model = data.drop(columns=['Name', 'Doctor', 'Hospital', 'Insurance Provider'])

# Encode categorical variables
categorical_cols = data_model.select_dtypes(include='object').columns
data_model = pd.get_dummies(data_model, columns=categorical_cols, drop_first=True)

# Split the data into features and target
X = data_model.drop(columns='Test Results')
y = data_model['Test Results']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'naive_bayes_model.pkl')

# Streamlit app
def main():
    st.title("Healthcare Patient Health Prediction")
    st.write("Predict the test result based on Room Number, Date of Admission, and Date of Discharge.")

    # Input fields
    room_number = st.number_input("Enter Room Number:", min_value=0, step=1)
    date_of_admission = st.date_input("Enter Date of Admission:")
    date_of_discharge = st.date_input("Enter Date of Discharge:")

    # Check if the inputs match a patient record in the dataset
    if st.button("Predict Health Status"):
        patient_data = data[(data['Room Number'] == room_number) & 
                            (data['Date of Admission'] == str(date_of_admission)) & 
                            (data['Discharge Date'] == str(date_of_discharge))]
        
        if patient_data.empty:
            st.error("No patient found with the given information.")
        else:
            # Select features for prediction
            patient_features = patient_data.drop(columns=['Name', 'Date of Admission', 'Discharge Date', 'Doctor', 'Hospital', 'Insurance Provider', 'Test Results'])
            patient_features_encoded = pd.get_dummies(patient_features, drop_first=True).reindex(columns=X.columns, fill_value=0)
            
            # Load model and make prediction
            model = joblib.load('naive_bayes_model.pkl')
            prediction = model.predict(patient_features_encoded)
            
            # Decode prediction
            health_status = {1: "Normal", 0: "Inconclusive", 2: "Abnormal"}
            result = health_status.get(prediction[0], "Unknown")
            
            # Display patient information and prediction
            st.subheader("Patient Information")
            patient_info = patient_data[['Name', 'Age', 'Gender', 'Blood Type', 'Medical Condition']]
            patient_info['Predicted Test Result'] = result
            st.write(patient_info)

if __name__ == "__main__":
    main()
