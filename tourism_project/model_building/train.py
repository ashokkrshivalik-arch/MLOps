
import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# 1. Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEPLOYMENT_DIR = os.path.join(PROJECT_DIR, "deployment")

os.makedirs(DEPLOYMENT_DIR, exist_ok=True)

# 2. Load datasets
Xtrain = pd.read_csv(os.path.join(DATA_DIR, "Xtrain.csv"))
Xtest = pd.read_csv(os.path.join(DATA_DIR, "Xtest.csv"))

ytrain = pd.read_csv(
    os.path.join(DATA_DIR, "ytrain.csv")
).squeeze("columns")

ytest = pd.read_csv(
    os.path.join(DATA_DIR, "ytest.csv")
).squeeze("columns")

print("Datasets loaded successfully.")

# 3. Define feature groups
numeric_cols = [
    "Age", "CityTier", "DurationOfPitch",
    "NumberOfPersonVisiting", "NumberOfFollowups",
    "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_cols = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation"
]

# 4. Preprocessing
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_cols),
    (
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        categorical_cols
    ),
    remainder="drop"
)

# 5. Handle class imbalance
class_counts = ytrain.value_counts()

class_weight = class_counts[0] / class_counts[1]

# 6. Create model pipeline
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)

# 7. Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [75, 100],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__colsample_bytree": [0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.5, 0.6],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.5, 0.6]
}

# 8. Configure MLflow
mlflow.set_tracking_uri(
    "file://" + os.path.join(PROJECT_DIR, "mlruns")
)

mlflow.set_experiment("Tourism_XGBoost_Experiment")

# 9. Train and evaluate
with mlflow.start_run(
    run_name="Optimized_XGB_Model_PROD_Standalone"
):

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy"
    )

    grid_search.fit(Xtrain, ytrain)

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])

            mlflow.log_metric(
                "mean_test_score",
                results["mean_test_score"][i]
            )

            mlflow.log_metric(
                "std_test_score",
                results["std_test_score"][i]
            )

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    # Training predictions
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]

    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)

    # Testing predictions
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)

    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True,
        zero_division=0
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1_score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1_score": test_report["1"]["f1-score"],
        "test_roc_auc": roc_auc_score(
            ytest,
            y_pred_test_proba
        )
    }

    mlflow.log_metrics(metrics)
    mlflow.log_metric(
        "classification_threshold",
        classification_threshold
    )

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print("\nModel Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 10. Save model
    MODEL_PATH = os.path.join(
        DEPLOYMENT_DIR,
        "best_tourism_model_v1.joblib"
    )

    joblib.dump(best_model, MODEL_PATH)

    print("\nModel saved successfully!")
    print("Model location:", MODEL_PATH)
