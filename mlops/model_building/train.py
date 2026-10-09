# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# for model serialization
import joblib

# 1. Load data 
Xtrain = pd.read_csv("Xtrain.csv")
Xtest  = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest  = pd.read_csv("ytest.csv").squeeze()

# ytrain and ytest are loaded as DataFrames, convert to Series if they contain only one column
ytrain = ytrain.squeeze()
ytest = ytest.squeeze()
print("Datasets loaded successfully.")

# 2. Define Feature Groups
numeric_cols = [
        'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 
        'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 
        'Passport', 'PitchSatisfactionScore', 'OwnCar', 
        'NumberOfChildrenVisiting', 'MonthlyIncome'
        ]


categorical_cols = [
        'TypeofContact', 'Occupation', 'Gender', 
        'ProductPitched', 'MaritalStatus', 'Designation'
        ]

# 3. Create Preprocessor & Pipeline
"""
preprocessor = make_column_transformer(
  (StandardScaler(), numeric_cols),
  (OneHotEncoder(handle_unknown='ignore'), categorical_cols)
  )"""
  
preprocessor = make_column_transformer(
        (StandardScaler(), numeric_cols),
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        remainder='drop' # Drops any column not explicitly mentioned
    )

# 4. Set the class weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# 5. Pipeline & Grid Search Setup
# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

model_pipeline = make_pipeline(preprocessor, xgb_model)

#6. Define hyperparameter grid
param_grid = {
        'xgbclassifier__n_estimators': [ 75, 100],
        'xgbclassifier__max_depth': [3, 4],
        'xgbclassifier__colsample_bytree': [0.5, 0.6],
        'xgbclassifier__colsample_bylevel': [0.5, 0.6],
        'xgbclassifier__learning_rate': [0.05, 0.1],
        'xgbclassifier__reg_lambda': [0.5, 0.6],
    }

#7. Execute Experiment

with mlflow.start_run(run_name="Optimized_XGB_Model_PROD_Standalone"):
  grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
  grid_search.fit(Xtrain, ytrain)

  # Log all parameter combinations and their mean test scores
  results = grid_search.cv_results_
  for i in range(len(results['params'])):
    param_set = results['params'][i]
    mean_score = results['mean_test_score'][i]
    std_score = results['std_test_score'][i]

  # Log each combination as a separate MLflow run
    with mlflow.start_run(nested=True):
      mlflow.log_params(param_set)
      mlflow.log_metric("mean_test_score", mean_score)
      mlflow.log_metric("std_test_score", std_score)

  # Log best parameters separately in main run
  mlflow.log_params(grid_search.best_params_)

  # Store and evaluate the best model
  best_model = grid_search.best_estimator_

  classification_threshold = 0.45


  #8. Comprehensive Metrics Logging

  y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
  y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

  y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
  y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

  train_report = classification_report(ytrain, y_pred_train, output_dict=True)
  test_report = classification_report(ytest, y_pred_test, output_dict=True)


  y_pred = best_model.predict(Xtest)
  y_proba = best_model.predict_proba(Xtest)[:, 1] # Needed for ROC-AUC

  # Log the metrics for the best model
  metrics = {
            "train_accuracy": train_report['accuracy'],
            "train_precision": train_report['1']['precision'],
            "train_recall": train_report['1']['recall'],
            "train_f1-score": train_report['1']['f1-score'],
            "test_accuracy": test_report['accuracy'],
            "test_precision": test_report['1']['precision'],
            "test_recall": test_report['1']['recall'],
            "test_f1-score": test_report['1']['f1-score'],
            "test_roc_auc": roc_auc_score(ytest, y_proba)
  }

  # Log the metrics for the best model
  mlflow.log_metrics(metrics)

  # Print all metrics
  print("Logged Metrics:")
  for key, value in metrics.items():
      print(f"  {key}: {value:.4f}"

# Save next to app.py so the Streamlit app can load it directly
joblib.dump(best_model, "deployment/best_medical_insurance_model_v1.joblib")
print("Model saved to deployment/best_medical_insurance_model_v1.joblib")
