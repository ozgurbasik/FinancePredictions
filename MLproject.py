import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

def load_and_clean_data(file_path):
    data = pd.read_csv(file_path, delimiter=';')
    data['y'] = data['y'].map({'yes': 1, 'no': 0})
    return data

def preprocess_data(data):
    # Define numerical columns
    numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                         'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                         'euribor3m', 'nr.employed']
    
    # Impute missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    data[numerical_columns] = imputer.fit_transform(data[numerical_columns])
    
    # One-hot encoding for selected categorical features
    one_hot_columns = ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']
    data = pd.get_dummies(data, columns=one_hot_columns, drop_first=True)
    
    # Ordinal encoding for selected categorical features
    ordinal_columns = ['default', 'housing', 'loan']
    ordinal_mapping = {'no': 0, 'yes': 1, 'unknown': 2}
    for col in ordinal_columns:
        data[col] = data[col].map(ordinal_mapping)
    
    # Feature scaling
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data, scaler

def select_features(data):
    X = data.drop(columns=['y'])
    y = data['y']
    return X, y

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
    print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
    print(f'F1 Score: {f1_score(y_test, y_pred)}')
    return model

def tune_hyperparameters(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print(f'Best Parameters: {grid_search.best_params_}')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    y_pred = best_model.predict(X_test)
    print(f'Tuned Model F1 Score: {f1_score(y_test, y_pred)}')
    return best_model, grid_search

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

def plot_validation_curve(X, y):
    param_range = np.arange(1, 11)
    train_scores, test_scores = validation_curve(
        RandomForestClassifier(), X, y, param_name="max_depth", param_range=param_range,
        cv=3, scoring="accuracy", n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(param_range, train_scores_mean, label="Training score", color="r")
    ax.plot(param_range, test_scores_mean, label="Cross-validation score", color="g")

    ax.fill_between(param_range, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.2, color="r")
    ax.fill_between(param_range, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.2, color="g")

    ax.set_title("Validation Curve with RandomForest")
    ax.set_xlabel("Max Depth")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc="best")
    st.pyplot(fig)

def load_model():
    return joblib.load('best_model.pkl')

def plot_mean_scores(grid_search):
    params = grid_search.cv_results_['params']
    mean_test_scores = grid_search.cv_results_['mean_test_score']
    
    param_values = [param['max_depth'] for param in params]  # Assuming 'max_depth' is one of the tuned parameters
    
    plt.plot(param_values, mean_test_scores, label='Mean Test Score')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Score')
    plt.title('Mean Test Score')
    plt.legend()
    st.pyplot(plt.gcf())


def plot_accuracy_vs_complexity(X_train, X_test, y_train, y_test):
    max_depth_values = [10, 20, 30]  # Change as per your tuned hyperparameters
    train_accuracies = []
    test_accuracies = []

    for max_depth in max_depth_values:
        model = RandomForestClassifier(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        train_accuracies.append(model.score(X_train, y_train))
        test_accuracies.append(model.score(X_test, y_test))

    plt.plot(max_depth_values, train_accuracies, label='Training Accuracy')
    plt.plot(max_depth_values, test_accuracies, label='Test Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Model Complexity')
    plt.legend()
    st.pyplot()

def plot_feature_importance(model, feature_names):
    feature_importances = model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    sorted_feature_importances = [feature_importances[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust the figure size as needed
    ax.bar(range(len(feature_importances)), sorted_feature_importances)
    ax.set_xlabel('Feature')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importance')
    ax.set_xticks(np.arange(len(sorted_feature_names)))  # Set ticks at each feature
    ax.set_xticklabels(sorted_feature_names, rotation=90)  # Rotate labels for better readability
    fig.tight_layout()  # Adjust layout to prevent clipping of labels
    st.pyplot(fig)

if __name__ == "__main__":
    data = load_and_clean_data('C:/Users/ozgur/Desktop/Documents/Code Library/0. Projects/MLFinancePredictions/bank-additional.csv')
    data, scaler = preprocess_data(data)
    X, y = select_features(data)
    best_model, grid_search = tune_hyperparameters(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    evaluate_model(best_model, X_test, y_test)
    plot_validation_curve(X, y)
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    st.title("Bank Term Deposit Prediction")

    # Predefined set of input data
    input_data = np.array([[35, 'admin.', 'married', 'university.degree', 'no', 'no', 'no', 'cellular', 'may', 'mon',
                            150, 3, 999, 0, 'nonexistent', 1.1, 93.994, -36.4, 4.857, 5191]])

    # Columns corresponding to the input data
    columns = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week',
               'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
               'cons.conf.idx', 'euribor3m', 'nr.employed']

    input_df = pd.DataFrame(input_data, columns=columns)

    # Define numerical columns (should match those defined in preprocess_data)
    numerical_columns = ['age', 'duration', 'campaign', 'pdays', 'previous', 
                         'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
                         'euribor3m', 'nr.employed']

    # Encode categorical features
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    ordinal_columns = ['default', 'housing', 'loan']
    ordinal_mapping = {'no': 0, 'yes': 1, 'unknown': 2}
    
    for column in ordinal_columns:
        input_df[column] = input_df[column].map(ordinal_mapping)

    input_df = pd.get_dummies(input_df, columns=[col for col in categorical_columns if col not in ordinal_columns], drop_first=True)
    
    # Align input data with training data features
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Scale numerical features
    numerical_features = input_df[numerical_columns]
    numerical_features_scaled = scaler.transform(numerical_features)
    
    input_df[numerical_columns] = numerical_features_scaled

   
        
    # Display confusion matrix and classification report
    st.subheader("Model Evaluation on Test Data")
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
        
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f'{val}', ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(fig)

    # Plot mean test scores and mean training scores
    st.subheader("Mean Test and Training Scores")
    plot_mean_scores(grid_search)

    # Plot accuracy vs. model complexity
    st.subheader("Accuracy vs. Model Complexity")
    plot_accuracy_vs_complexity(X_train, X_test, y_train, y_test)

    # Plot feature importance
    st.subheader("Feature Importance")
    plot_feature_importance(best_model, X.columns)

# streamlit run "C:\Users\ozgur\Desktop\Documents\Code Library\0. Projects\MLFinancePredictions\MLproject.py"