import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def load_and_clean_and_preprocess_and_engineer_data(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path, delimiter=';')
    
    # Perform mean imputation for missing values in numeric columns
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    # Map target variable to numerical values
    data['y'] = data['y'].map({'yes': 1, 'no': 0})
    
        # Drop the 'duration' column
    data = data.drop(columns=['duration'])

    # One-hot encoding for selected categorical features
    one_hot_columns = ['job', 'marital', 'default', 'housing', 'loan', 'month', 'day_of_week', 'contact', 'poutcome','education']
    data = pd.get_dummies(data, columns=one_hot_columns, drop_first=True)

    
    # Define columns to scale
    columns_to_scale = ['age', 'pdays', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 
                        'cons.conf.idx', 'euribor3m', 'nr.employed']
    
    # Feature scaling
    scaler = StandardScaler()
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
    
    # Check if columns are correctly named after one-hot encoding
    intersection_columns = [col for col in ['job', 'campaign'] if col in data.columns]
    if len(intersection_columns) == 2:
        data['job_campaign'] = data['job'] * data['campaign']
    
    # Generate additional intersection features
    if 'campaign' in data.columns and 'age' in data.columns:
        data['campaign++++age'] = data['campaign'] * data['age']
    if 'campaign' in data.columns and 'euribor3m' in data.columns:
        data['campaign++++euribor3m'] = data['campaign'] * data['euribor3m']
    if 'campaign' in data.columns and 'nr.employed' in data.columns:
        data['campaign++++nr.employed'] = data['campaign'] * data['nr.employed']
    if 'campaign' in data.columns and 'pdays' in data.columns:
        data['campaign++++pdays'] = data['campaign'] * data['pdays']
    
    return data

def plot_confusion_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    specificity = cm[1,1] / (cm[1,0] + cm[1,1])  # TN / (TN + FP)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.figtext(0.2, -0.1, f'Accuracy: {accuracy:.2f}', ha='left')
    plt.figtext(0.2, -0.15, f'Precision: {precision:.2f}', ha='left')
    plt.figtext(0.2, -0.2, f'Recall: {recall:.2f}', ha='left')
    plt.figtext(0.2, -0.25, f'F1 Score: {f1:.2f}', ha='left')
    plt.figtext(0.2, -0.3, f'Specificity: {specificity:.2f}', ha='left')
    st.pyplot(plt)

def plot_accuracy_vs_max_depth(X_train, X_test, y_train, y_test, max_depths):
        
    train_accuracies = []
    test_accuracies = []

    for depth in max_depths:
        # Train decision tree classifier
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)

        # Predict and calculate training accuracy
        y_train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_accuracies.append(train_accuracy)

        # Predict and calculate testing accuracy
        y_test_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_accuracies.append(test_accuracy)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(max_depths, train_accuracies, marker='o', linestyle='-', color='blue', label='Training Accuracy')
    plt.plot(max_depths, test_accuracies, marker='o', linestyle='-', color='green', label='Testing Accuracy')
    plt.title('Accuracy vs Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.xticks(max_depths)
    plt.legend()
    st.pyplot(plt)

def plot_feature_importance(model, feature_names):
    if isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_feature_names = [feature_names[i] for i in indices]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), sorted_feature_names, rotation=90)
        plt.title('Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        st.pyplot(plt)
    else:
        st.write("Feature importances plot is not available for this model.")

def train_models_1(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=32)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    model = GridSearchCV(RandomForestClassifier(random_state=32), param_grid, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix for Random Forest Classifier')
    plot_accuracy_vs_max_depth(X_train, X_test, y_train, y_test, range(1, 21))
    plot_feature_importance(model.best_estimator_, feature_names)
    
    

def train_models_2(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=32)
    
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }
    
    model = GridSearchCV(LogisticRegression(random_state=32), param_grid, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix for Logistic Regression')
    plot_accuracy_vs_max_depth(X_train, X_test, y_train, y_test, range(1, 21))
    plot_feature_importance(model.best_estimator_, feature_names)

def train_models_3(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=32)
    
    param_grid = {
        'C': [0.1, 1, 10, 20],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
    }
    
    model = GridSearchCV(SVC(random_state=32), param_grid, cv=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix for Support Vector Classifier')
    plot_accuracy_vs_max_depth(X_train, X_test, y_train, y_test, range(1, 21))
    plot_feature_importance(model.best_estimator_, feature_names)
    


# Function to center text in Streamlit
def centered_title(text):
    st.markdown(f"<h1 style='text-align: center;'>{text}</h1>", unsafe_allow_html=True)

# Function to center text in Streamlit
def centered_text(text):
    st.markdown(f"<h4 style='text-align: center;'>{text}</h4>", unsafe_allow_html=True)

def centered_text2(text):
    st.markdown(f"<h6 style='text-align: center;'>{text}</h6>", unsafe_allow_html=True)

def centered_text3(text):
    st.markdown(f"<h10 style='text-align: center;'>{text}</h10>", unsafe_allow_html=True)



if __name__ == "__main__":

    centered_title("Bank Term Deposit Prediction")
    centered_text("Choose an training algorithm to proceed")
    centered_text2("Select a CVS File with: ")
    centered_text3("Input Var:")
    centered_text3("age, job, marital, education, default, housing, loan, contact, month, day_of_week," 
    "duration, campaign, pdays, previous, poutcome, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed")
    centered_text3("Output Var:")
    centered_text3("y - has the client subscribed a term deposit? (binary: yes, no)")
# Use Streamlit file uploader for user-friendly file input
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = load_and_clean_and_preprocess_and_engineer_data(uploaded_file)
    X = data.drop(columns=['y'])
    y = data['y']

    # Add buttons next to each other
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button('Model 1 Random Forest Classifier'):
            feature_names = X.columns.tolist()
            model_1 = train_models_1(X, y , feature_names)
    
    with col2:
        if st.button('Model 2 Logistic Regression'):
            feature_names = X.columns.tolist()
            model_2 = train_models_2(X, y , feature_names)
    
    with col3:
        if st.button('Model 3 Support Vector Classifier'):
            feature_names = X.columns.tolist()
            model_3 = train_models_3(X, y , feature_names)


# streamlit run "C:\Users\ozgur\Desktop\Documents\Code Library\0. Projects\MLFinancePredictions\MLproject2.py"