# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Imports for Model Training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# Streamlit app layout
def anal():
    st.title("SAGE")
    st.header("Data Analysis")

    # Upload dataset in the sidebar
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("### Preview of Original Dataset", df)

        # Main page navigation options
        st.header("Select a Section to Proceed")
        option = st.selectbox("Choose an option", ("Dataset Overview", "Data Cleaning", "Model Training"))

        profile = ProfileReport(df, title='Profile Report')
        profile.to_file(r"report.html")

        # Data Cleaning Section
        if option == "Dataset Overview":
            st.subheader("Dataset Overview")
            with open("report.html", "r", encoding='utf-8') as html_file:
                source = html_file.read()
            components.html(source, height=1000, scrolling=True)

        # Data Cleaning Section
        elif option == "Data Cleaning":
            st.subheader("Data Cleaning Options")

            missing_cols = df.columns[df.isnull().any()]

            # Missing Values Handling
            if st.button("Show Columns with Missing Values"):
                # Create a DataFrame to display the columns and their respective missing value count
                missing_df = pd.DataFrame({
                    'Column': missing_cols,
                    'Missing Values': df[missing_cols].isnull().sum()
                }).reset_index(drop=True)
                
                # Display the DataFrame
                st.write(f"### Columns with missing values and their count:")
                st.dataframe(missing_df)
                
            handle_missing_option = st.selectbox("How to handle missing values?", 
                                                ("None", "Drop columns", "Impute (Mean)", "Impute (Median)", "Impute (Mode)"))
            
            apply_to_all = st.checkbox("Apply to all columns with missing values", value=True)
            
            if handle_missing_option == "Drop columns":
                if apply_to_all:
                    st.write(f"Dropping all columns with missing values.")
                    df = df.dropna(axis=1)
                else:
                    # Provide an option to select which columns to drop
                    columns_to_drop = st.multiselect("Select columns to drop", missing_cols.tolist())
                    if columns_to_drop:
                        st.write(f"Dropping selected columns: {columns_to_drop}")
                        df = df.drop(columns=columns_to_drop)
                    else:
                        st.write("No columns selected for dropping.")
                st.write("### Dataset after dropping columns with missing values", df)
            
            elif handle_missing_option == "Impute All":
                if st.button("Impute All Columns"):
                    if "Impute (Mean)" in st.session_state:
                        fill_value = df.mean()
                    elif "Impute (Median)" in st.session_state:
                        fill_value = df.median()
                    elif "Impute (Mode)" in st.session_state:
                        fill_value = df.mode().iloc[0]
                    
                    df.fillna(fill_value, inplace=True)
                    st.write(f"### Dataset after imputing {handle_missing_option.lower()}", df)

            elif handle_missing_option in ["Impute (Mean)", "Impute (Median)", "Impute (Mode)"]:
                st.write("Select the columns to impute")
                columns_to_impute = st.multiselect("Select Columns", missing_cols)
                
                if handle_missing_option == "Impute (Mean)":
                    for column in columns_to_impute:
                        df[column].fillna(df[column].mean(), inplace=True)
                
                elif handle_missing_option == "Impute (Median)":
                    for column in columns_to_impute:
                        df[column].fillna(df[column].median(), inplace=True)
                
                elif handle_missing_option == "Impute (Mode)":
                    for column in columns_to_impute:
                        df[column].fillna(df[column].mode()[0], inplace=True)
                
                st.write(f"### Dataset after imputing {handle_missing_option.lower()} for selected columns", df)

            # One-Hot Encoding
            st.write("### One-Hot Encoding Options")
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if len(categorical_cols) > 0:
                encode_cols = st.multiselect("Select Categorical Columns to One-Hot Encode", categorical_cols)
                if len(encode_cols) > 0:
                    df = pd.get_dummies(df, columns=encode_cols)
                    st.write("### Dataset after one-hot encoding", df)
            else:
                st.write("No categorical columns to encode.")
            
            # Duplicates Handling
            if st.button("Show Duplicates"):
                duplicates = df[df.duplicated()]
                if len(duplicates) > 0:
                    st.write(f"### Duplicates found: {len(duplicates)}")
                    st.write(duplicates)
                else:
                    st.write("No duplicates found.")
            
            if st.checkbox("Drop Duplicates"):
                df = df.drop_duplicates()
                st.write("### Dataset after dropping duplicates", df)
            
            st.subheader("Data Preprocessing")
            if st.button('Standardize Dataset'):
                target_column = st.selectbox("Select Target Variable", df.columns)
                standardize_features = st.checkbox("Standardize Features")
                if standardize_features:
                    scaler = StandardScaler()
                    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
                    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
                    st.write("### Dataset after standardization", df)
            
            # Export Options Section
            if st.button("Download Cleaned Data"):
                cleaned_data = df.to_csv(index=False)
                st.download_button("Download CSV", data=cleaned_data, file_name="cleaned_data.csv")

        # Model Training Section
        elif option == "Model Training":
            st.subheader("Model Training Options")

            # Input X and Y values for model training
            feature_cols = st.multiselect("Select Features (X)", df.columns.tolist(), default=df.columns.tolist())
            target_column = st.selectbox("Select Target Variable (Y)", df.columns)

            if len(feature_cols) > 0 and target_column:
                X = df[feature_cols]
                y = df[target_column]

                # Display train-test split overview
                st.write("### Train-Test Split Overview")
                test_size = st.slider("Test Size (Fraction)", 0.1, 0.5, 0.2, 0.05)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                st.write(f"Training Set Size: {X_train.shape[0]}")
                st.write(f"Testing Set Size: {X_test.shape[0]}")

                # Standardize Features Option
                standardize_features = st.checkbox("Standardize Features")
                if standardize_features:
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)
                    st.write("### Dataset after Standardization")
                    st.write("Training Data:", X_train[:5])
                    st.write("Testing Data:", X_test[:5])

                # Model Selection
                model_type = st.selectbox("Select Model", ["Random Forest", "Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors", "Decision Tree", "Naive Bayes", "Gradient Boosting"])

                # Train and Evaluate Model
                if st.button("Train Model"):
                    if model_type == "Random Forest":
                        model = RandomForestClassifier() if y.dtype == 'object' else RandomForestRegressor()
                    elif model_type == "Logistic Regression":
                        model = LogisticRegression()
                    elif model_type == "Support Vector Machine":
                        model = SVC() if y.dtype == 'object' else SVR()
                    elif model_type == "K-Nearest Neighbors":
                        model = KNeighborsClassifier() if y.dtype == 'object' else KNeighborsRegressor()
                    elif model_type == "Decision Tree":
                        model = DecisionTreeClassifier() if y.dtype == 'object' else DecisionTreeRegressor()
                    elif model_type == "Naive Bayes":
                        model = GaussianNB() if y.dtype == 'object' else GaussianNB()
                    elif model_type == "Gradient Boosting":
                        model = GradientBoostingClassifier() if y.dtype == 'object' else GradientBoostingRegressor()

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Metrics Scoring
                    if y.dtype == 'object':
                        # Classification Metrics
                        accuracy = accuracy_score(y_test, y_pred)
                        st.write(f"### Model Accuracy: {accuracy*100:.2f}%")
                        st.write("### Confusion Matrix")
                        st.write(confusion_matrix(y_test, y_pred))
                    else:
                        # Regression Metrics
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        st.write(f"### Mean Squared Error: {mse:.2f}")
                        st.write(f"### R-squared: {r2:.2f}")

            else:
                st.write("Please select features and target variable for model training.")


    else:
        st.write("Please upload a dataset to begin!")
