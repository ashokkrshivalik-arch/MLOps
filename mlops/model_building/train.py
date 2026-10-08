
# ============================================================
# Tourism Project - XGBoost Model Training with MLflow Tracking
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
import mlflow

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score
)

import xgboost as xgb


# ------------------------------------------------------------
# 1. Load Training and Test Data
# ------------------------------------------------------------

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")

ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()


# ------------------------------------------------------------
# 2. Define Feature Groups
# ------------------------------------------------------------

numeric_cols = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

categorical_cols = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]


# ------------------------------------------------------------
# 3. Create Preprocessor
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 4. Handle Class Imbalance
# ------------------------------------------------------------

class_weight = (
    ytrain.value_counts()[0]
    / ytrain.value_counts()[1]
)

print(f"Scale Pos Weight: {class_weight:.4f}")


# ------------------------------------------------------------
# 5. Create XGBoost Model and Pipeline
# ------------------------------------------------------------

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


# ------------------------------------------------------------
# 6. Hyperparameter Grid
# ------------------------------------------------------------

param_grid = {
    "xgbclassifier__n_estimators": [75, 100],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__colsample_bytree": [0.5, 0.6],
    "xgbclassifier__colsample_bylevel": [0.5, 0.6],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__reg_lambda": [0.5, 0.6],
}


# ------------------------------------------------------------
# 7. Grid Search + MLflow Experiment
# ------------------------------------------------------------

with mlflow.start_run(
    run_name="Optimized_XGB_Model_PROD_Standalone"
):

    grid_search = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="accuracy"
    )

    grid_search.fit(Xtrain, ytrain)

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(
        f"\nBest Cross-Validation Score: "
        f"{grid_search.best_score_:.4f}"
    )


    # --------------------------------------------------------
    # Log all parameter combinations as nested MLflow runs
    # --------------------------------------------------------

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        param_set = results["params"][i]
        mean_score = results["mean_test_score"][i]
        std_score = results["std_test_score"][i]

        with mlflow.start_run(
            run_name=f"GridSearch_Run_{i + 1}",
            nested=True
        ):

            mlflow.log_params(param_set)

            mlflow.log_metric(
                "mean_test_score",
                mean_score
            )

            mlflow.log_metric(
                "std_test_score",
                std_score
            )


    # --------------------------------------------------------
    # 8. Best Model
    # --------------------------------------------------------

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    classification_threshold = 0.45

    mlflow.log_param(
        "classification_threshold",
        classification_threshold
    )


    # --------------------------------------------------------
    # 9. Predictions
    # --------------------------------------------------------

    # Training predictions
    y_pred_train_proba = best_model.predict_proba(
        Xtrain
    )[:, 1]

    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)


    # Test predictions
    y_pred_test_proba = best_model.predict_proba(
        Xtest
    )[:, 1]

    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)


    # --------------------------------------------------------
    # 10. Classification Reports
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 11. Metrics
    # --------------------------------------------------------

    metrics = {

        "train_accuracy":
            train_report["accuracy"],

        "train_precision":
            train_report["1"]["precision"],

        "train_recall":
            train_report["1"]["recall"],

        "train_f1_score":
            train_report["1"]["f1-score"],

        "test_accuracy":
            test_report["accuracy"],

        "test_precision":
            test_report["1"]["precision"],

        "test_recall":
            test_report["1"]["recall"],

        "test_f1_score":
            test_report["1"]["f1-score"],

        "test_roc_auc":
            roc_auc_score(
                ytest,
                y_pred_test_proba
            )
    }


    # --------------------------------------------------------
    # 12. Log Metrics
    # --------------------------------------------------------

    mlflow.log_metrics(metrics)

    print("\nLogged Metrics:")

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


    # --------------------------------------------------------
    # 13. Save Production Model
    # --------------------------------------------------------

    model_dir = (
        "mls10-wellness_tourism_mlops/"
        "model_building"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        model_dir,
        "productionmodel.joblib"
    )

    joblib.dump(
        best_model,
        model_path
    )


    # --------------------------------------------------------
    # 14. Log Model Artifact
    # --------------------------------------------------------

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print(
        f"\nModel saved as artifact at: "
        f"{model_path}"
    )


print("\nModel Training & Tracking Complete.")

print("\nFinal Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
