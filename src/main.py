import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

# Data generation
np.random.seed(42)
num_samples = 5000
age = np.random.randint(18, 70, num_samples)
gender = np.random.choice(['Male', 'Female'], num_samples)
previous_response = np.random.choice(['Positive', 'Neutral', 'Negative'], num_samples)
customer_satisfaction = np.random.randint(1, 6, num_samples)
time_spent_on_survey = np.random.randint(1, 30, num_samples)
purchase_frequency = np.random.randint(1, 10, num_samples)
annual_income = np.random.randint(20000, 100000, num_samples)
satisfaction_score = np.random.uniform(1, 5, num_samples)

df = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Previous_Response': previous_response,
    'Customer_Satisfaction': customer_satisfaction,
    'Time_Spent_on_Survey': time_spent_on_survey,
    'Purchase_Frequency': purchase_frequency,
    'Annual_Income': annual_income,
    'Satisfaction_Score': satisfaction_score
})

# Data preprocessing
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

X = df.drop('Satisfaction_Score', axis=1)
y = df['Satisfaction_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training and evaluation
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{model.__class__.__name__} MSE: {mse:.2f}')
    print(f'{model.__class__.__name__} R2 Score: {r2:.2f}')

lr_model = LinearRegression()
train_and_evaluate_model(lr_model, X_train, y_train, X_test, y_test)

gbr_model = GradientBoostingRegressor(random_state=42)
train_and_evaluate_model(gbr_model, X_train, y_train, X_test, y_test)

# Hyperparameter tuning with GridSearchCV
def grid_search(model, param_grid, X_train, y_train):
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

param_grid_gbr = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

best_gbr = grid_search(GradientBoostingRegressor(random_state=42), param_grid_gbr, X_train_scaled, y_train)
train_and_evaluate_model(best_gbr, X_train_scaled, y_train, X_test_scaled, y_test)

param_grid_xgb = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

best_xgb = grid_search(xgb.XGBRegressor(random_state=42), param_grid_xgb, X_train_scaled, y_train)
train_and_evaluate_model(best_xgb, X_train_scaled, y_train, X_test_scaled, y_test)

lgb_model = lgb.LGBMRegressor(random_state=42)
train_and_evaluate_model(lgb_model, X_train, y_train, X_test, y_test)

# Random Forest with GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30],
    'criterion': ['gini', 'entropy']
}

best_rf = grid_search(RandomForestClassifier(random_state=42), param_grid_rf, X_train_scaled, y_train)
best_rf.fit(X_train_scaled, y_train)
y_pred_rf = best_rf.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf * 100:.2f}%')
print(classification_report(y_test, y_pred_rf))

# Personalize survey function
def personalize_survey(responses, label_encoders, model):
    for column, encoder in label_encoders.items():
        if column in responses:
            responses[column] = encoder.transform([responses[column]])[0]
    responses_df = pd.DataFrame([responses])
    prediction = model.predict(responses_df)
    return label_encoders['Next_Question'].inverse_transform(prediction)[0]

# Example usage
responses = {
    'Age': 25,
    'Gender': 'Female',
    'Previous_Response': 'Positive',
    'Customer_Satisfaction': 4,
    'Time_Spent_on_Survey': 15,
    'Purchase_Frequency': 5,
    'Annual_Income': 60000
}
# next_question = personalize_survey(responses, label_encoders, best_rf)
# print(f'Next Question: {next_question}')
