import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("CGPA Prediction using Linear Regression")

# Upload CSV data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except:
        df = pd.read_excel(uploaded_file)

    # Drop the 'Timestamp' column
    
    

    academic_year_map = {
        '1st Year': 1,
        '2nd Year': 2,
        '3rd Year': 3,
        '4th Year': 4,
        # add other years as needed
    }

    hours_online_map = {
        '1-3 hours': 1,
        '4-6 hours': 2,
        'More than 6 hours': 3,
        'Less than 1 hour': 0
    }

    time_management_map = {
        'Very Poorly': 1,
        'Poorly': 2,
        'Neutral': 3,
        'Well': 4,
        'Very Well': 5
    }

    gpa_map = {
        '2.6 - 3.0': 2.8,
        '2.0 - 2.5': 2.25,
        '3.1 - 3.5': 3.3,
        '3.6 - 4.0': 3.8,
    }

    df['What is your current Academic Year? (University)'] = df['What is your current Academic Year? (University)'].map(academic_year_map)
    df['How much time do you spend online for academic activities per day?'] = df['How much time do you spend online for academic activities per day?'].map(hours_online_map)
    df['How much time do you spend online for recreational activities per day?'] = df['How much time do you spend online for recreational activities per day?'].map(hours_online_map)
    df['On a scale of 1-5, how effectively do you manage your time between online academic and recreational activities?'] = df['On a scale of 1-5, how effectively do you manage your time between online academic and recreational activities? '].map(time_management_map)
    df['What is your current GPA?'] = df['What is your current GPA?'].map(gpa_map)

    df = pd.get_dummies(df, columns=['What is your major?', 'What is your primary reason for using the internet?'])


    df.drop(['On a scale of 1-5, how effectively do you manage your time between online academic and recreational activities? ',
        ' How many hours do you spend online per day?'], axis=1, inplace=True)
    df.dropna(inplace=True)
    
    
    df_cleaned = df.drop(columns=['Timestamp'])

    # Define X (features) and y (target variable)
    X = df_cleaned.drop(columns=['What is your current GPA?'])  # Independent variables
    y = df_cleaned['What is your current GPA?']  # Dependent variable (CGPA)
#     X.drop("GPA", axis=1, inplace=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the Linear Regression model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display performance metrics
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R-squared: {r2}")

    # Show the coefficients of each feature
    coefficients = pd.DataFrame(reg.coef_, X.columns, columns=['Coefficient'])

    st.subheader("Factors Affecting CGPA")
    st.dataframe(coefficients)

    # Plotting the coefficients
    st.subheader("Feature Coefficients Bar Plot")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=coefficients.index, y=coefficients['Coefficient'], palette="coolwarm")
    plt.xticks(rotation=90)
    plt.title('Feature Coefficients for Predicting CGPA')
    plt.xlabel('Features')
    plt.ylabel('Coefficient Value')
    st.pyplot(plt)

    # Plot actual vs predicted CGPA values
    st.subheader("Actual vs Predicted CGPA")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # 45-degree line
    plt.title('Actual vs Predicted CGPA')
    plt.xlabel('Actual CGPA')
    plt.ylabel('Predicted CGPA')
    st.pyplot(plt)

    # Plot residuals
    residuals = y_test - y_pred
    st.subheader("Residuals Plot")
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.7, color="purple")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted CGPA')
    plt.ylabel('Residuals (Actual - Predicted)')
    st.pyplot(plt)
