# ============================================================
# CAREMATRIX - RISK PREDICTION AGENT TRAINING
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import vitaldb

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION
# ============================================================

WINDOW_SECONDS = 300
SAMPLE_INTERVAL = 5
MAX_TRAINING_MINUTES = 60

N_CASES = 300
RANDOM_STATE = 42


VITAL_TRACKS = [
    "Solar8000/HR",
    "Solar8000/PLETH_SPO2",
    "Solar8000/RR",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Solar8000/NIBP_MBP",
    "Solar8000/BT"
]


VITAL_COLUMNS = [
    "HR",
    "SpO2",
    "RR",
    "SBP",
    "DBP",
    "MAP",
    "BT"
]


# ============================================================
# 2. LOAD CLINICAL DATA
# ============================================================

print("=" * 60)
print("LOADING VITALDB CLINICAL DATA")
print("=" * 60)

cases = pd.read_csv(
    "https://api.vitaldb.net/cases"
)

print(
    "Total VitalDB cases:",
    len(cases)
)


trks = pd.read_csv(
    "https://api.vitaldb.net/trks"
)


required_cases = (
    trks[
        trks["tname"].isin(VITAL_TRACKS)
    ]
    .groupby("caseid")["tname"]
    .nunique()
)


valid_case_ids = required_cases[
    required_cases == len(VITAL_TRACKS)
].index.tolist()


print(
    "Cases containing all 7 required tracks:",
    len(valid_case_ids)
)


# ============================================================
# 3. SELECT CASES
# ============================================================

available_cases = cases[
    cases["caseid"].isin(valid_case_ids)
].copy()


available_cases = available_cases.dropna(
    subset=["caseid"]
)


# Create outcome first so sampling can be stratified
available_cases["death_inhosp"] = pd.to_numeric(
    available_cases["death_inhosp"],
    errors="coerce"
).fillna(0)


available_cases["icu_days"] = pd.to_numeric(
    available_cases["icu_days"],
    errors="coerce"
).fillna(0)


available_cases["High Risk"] = (
    (available_cases["death_inhosp"] == 1) |
    (available_cases["icu_days"] > 0)
).astype(int)


# Select 300 cases while preserving the two outcome classes
if len(available_cases) > N_CASES:

    available_cases, _ = train_test_split(
        available_cases,
        test_size=len(available_cases) - N_CASES,
        random_state=RANDOM_STATE,
        stratify=available_cases["High Risk"]
    )


available_cases = available_cases.reset_index(
    drop=True
)


print(
    "Cases selected for training:",
    len(available_cases)
)


print("\nOutcome distribution:")
print(
    available_cases["High Risk"].value_counts()
)


# ============================================================
# 4. FEATURE EXTRACTION
# ============================================================

def extract_window_features(window_df):

    features = {}

    for vital in VITAL_COLUMNS:

        if vital not in window_df.columns:
            continue

        values = pd.to_numeric(
            window_df[vital],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        features[f"{vital}_mean"] = float(
            values.mean()
        )

        features[f"{vital}_min"] = float(
            values.min()
        )

        features[f"{vital}_max"] = float(
            values.max()
        )

        if len(values) > 1:

            features[f"{vital}_std"] = float(
                values.std()
            )

        else:

            features[f"{vital}_std"] = 0.0


        features[f"{vital}_latest"] = float(
            values.iloc[-1]
        )


        features[f"{vital}_change"] = float(
            values.iloc[-1] -
            values.iloc[0]
        )


        if len(values) > 1:

            x = np.arange(
                len(values)
            )

            try:

                slope = np.polyfit(
                    x,
                    values.values,
                    1
                )[0]

            except Exception:

                slope = 0.0

        else:

            slope = 0.0


        features[f"{vital}_slope"] = float(
            slope
        )


    return features


# ============================================================
# 5. LOAD PATIENT WINDOWS
# ============================================================

def create_patient_windows(case_id):

    try:

        data = vitaldb.load_case(
            int(case_id),
            VITAL_TRACKS,
            interval=SAMPLE_INTERVAL
        )

    except Exception as e:

        print(
            f"Could not load case {case_id}: {e}"
        )

        return []


    if data is None:
        return []


    # VitalDB returns a NumPy array in this environment
    if isinstance(data, np.ndarray):

        data = pd.DataFrame(
            data,
            columns=VITAL_COLUMNS
        )

        data.insert(
            0,
            "Time",
            np.arange(len(data)) *
            SAMPLE_INTERVAL
        )


    elif isinstance(data, pd.DataFrame):

        data = data.copy()

        if "Time" not in data.columns:

            data.insert(
                0,
                "Time",
                np.arange(len(data)) *
                SAMPLE_INTERVAL
            )

        rename_map = {

            "Solar8000/HR": "HR",

            "Solar8000/PLETH_SPO2":
                "SpO2",

            "Solar8000/RR":
                "RR",

            "Solar8000/NIBP_SBP":
                "SBP",

            "Solar8000/NIBP_DBP":
                "DBP",

            "Solar8000/NIBP_MBP":
                "MAP",

            "Solar8000/BT":
                "BT"
        }


        data = data.rename(
            columns=rename_map
        )


    else:

        print(
            f"Unexpected data type for case "
            f"{case_id}: {type(data)}"
        )

        return []


    for vital in VITAL_COLUMNS:

        if vital not in data.columns:

            data[vital] = np.nan


    data = data[
        ["Time"] + VITAL_COLUMNS
    ].copy()


    data["Time"] = pd.to_numeric(
        data["Time"],
        errors="coerce"
    )


    for vital in VITAL_COLUMNS:

        data[vital] = pd.to_numeric(
            data[vital],
            errors="coerce"
        )


    data = data.dropna(
        subset=["Time"]
    )


    data = data[
        data["Time"] <=
        MAX_TRAINING_MINUTES * 60
    ].copy()


    if len(data) == 0:
        return []


    windows = []


    max_time = float(
        data["Time"].max()
    )


    start_time = 0


    while (
        start_time +
        WINDOW_SECONDS <=
        max_time
    ):

        end_time = (
            start_time +
            WINDOW_SECONDS
        )


        window = data[
            (data["Time"] >= start_time) &
            (data["Time"] < end_time)
        ].copy()


        if len(window) > 0:

            features = (
                extract_window_features(
                    window
                )
            )


            if len(features) > 0:

                features["caseid"] = int(
                    case_id
                )

                features["window_start"] = (
                    start_time
                )

                features["window_end"] = (
                    end_time
                )

                windows.append(
                    features
                )


        start_time += WINDOW_SECONDS


    return windows


# ============================================================
# 6. CREATE WINDOW DATASET
# ============================================================

print("\n" + "=" * 60)
print("CREATING 5-MINUTE PATIENT WINDOWS")
print("=" * 60)


all_rows = []


for index, row in available_cases.iterrows():

    case_id = int(
        row["caseid"]
    )


    if (index + 1) % 25 == 0:

        print(
            f"Processed {index + 1}/"
            f"{len(available_cases)} cases"
        )


    windows = create_patient_windows(
        case_id
    )


    for window in windows:

        window["age"] = row.get(
            "age",
            np.nan
        )

        window["sex"] = row.get(
            "sex",
            np.nan
        )

        window["bmi"] = row.get(
            "bmi",
            np.nan
        )

        window["asa"] = row.get(
            "asa",
            np.nan
        )

        window["emop"] = row.get(
            "emop",
            np.nan
        )

        window["High Risk"] = int(
            row["High Risk"]
        )

        all_rows.append(
            window
        )


df = pd.DataFrame(
    all_rows
)


print("\nFinal window dataset:")
print(
    df.shape
)


if len(df) == 0:

    raise RuntimeError(
        "No patient windows were created."
    )


# ============================================================
# 7. PREPROCESS PATIENT FEATURES
# ============================================================

if "sex" in df.columns:

    df["sex"] = (
        df["sex"]
        .astype(str)
        .str.upper()
        .map({
            "M": 1,
            "F": 0
        })
    )


for col in [
    "age",
    "bmi",
    "asa",
    "emop"
]:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# 8. CASE-LEVEL TRAIN / TEST SPLIT
# ============================================================

case_ids = df[
    "caseid"
].unique()


case_labels = (
    df.groupby("caseid")
    ["High Risk"]
    .first()
)


train_cases, test_cases = train_test_split(
    case_ids,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=case_labels
)


train_df = df[
    df["caseid"].isin(train_cases)
].copy()


test_df = df[
    df["caseid"].isin(test_cases)
].copy()


print(
    "\nTraining windows:",
    train_df.shape
)

print(
    "Testing windows:",
    test_df.shape
)


print(
    "\nTraining case outcomes:"
)

print(
    train_df.groupby(
        "High Risk"
    )["caseid"].nunique()
)


print(
    "\nTesting case outcomes:"
)

print(
    test_df.groupby(
        "High Risk"
    )["caseid"].nunique()
)


# ============================================================
# 9. FEATURES
# ============================================================

IGNORE_COLUMNS = [
    "caseid",
    "window_start",
    "window_end",
    "High Risk"
]


ML_FEATURES = [
    col
    for col in train_df.columns
    if col not in IGNORE_COLUMNS
]


X_train_full = train_df[
    ML_FEATURES
].copy()


y_train_full = train_df[
    "High Risk"
].astype(int)


X_test = test_df[
    ML_FEATURES
].copy()


y_test = test_df[
    "High Risk"
].astype(int)


print(
    "\nNumber of ML features:",
    len(ML_FEATURES)
)


print(
    "\nHigh Risk training windows:",
    y_train_full.sum()
)


print(
    "Low Risk training windows:",
    (y_train_full == 0).sum()
)


# ============================================================
# 10. INTERNAL VALIDATION SPLIT
# ============================================================

train_case_labels = (
    train_df.groupby("caseid")
    ["High Risk"]
    .first()
)


model_train_cases, validation_cases = (
    train_test_split(
        train_cases,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=train_case_labels
    )
)


model_train_df = train_df[
    train_df["caseid"].isin(
        model_train_cases
    )
].copy()


validation_df = train_df[
    train_df["caseid"].isin(
        validation_cases
    )
].copy()


X_model_train = model_train_df[
    ML_FEATURES
].copy()


y_model_train = model_train_df[
    "High Risk"
].astype(int)


X_validation = validation_df[
    ML_FEATURES
].copy()


y_validation = validation_df[
    "High Risk"
].astype(int)


print(
    "\nModel-training windows:",
    X_model_train.shape
)


print(
    "Validation windows:",
    X_validation.shape
)


print(
    "\nValidation class distribution:"
)

print(
    y_validation.value_counts()
)


# ============================================================
# 11. PREPROCESSOR
# ============================================================

numeric_features = (
    X_model_train.columns.tolist()
)


preprocessor = ColumnTransformer(
    transformers=[

        (
            "numeric",

            Pipeline([

                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),

                (
                    "scaler",
                    StandardScaler()
                )

            ]),

            numeric_features
        )
    ]
)


X_model_train_processed = (
    preprocessor.fit_transform(
        X_model_train
    )
)


X_validation_processed = (
    preprocessor.transform(
        X_validation
    )
)


X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


# ============================================================
# 12. MODELS
# ============================================================

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),


    "Decision Tree":

        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),


    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),


    "Extra Trees":

        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),


    "Gradient Boosting":

        GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),


    "HistGradientBoosting":

        HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),


    "XGBoost":

        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )
}


# ============================================================
# 13. THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1

    for threshold in np.arange(
        0.10,
        0.91,
        0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)


        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )


        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )


        # Prefer F1, with recall as a tie-breaker
        score = (
            f1,
            recall
        )


        best_score = (
            best_f1,
            -1
        )


        if score > best_score:

            best_f1 = f1
            best_threshold = float(
                threshold
            )


    return best_threshold


# ============================================================
# 14. TRAIN MODELS
# ============================================================

trained_models = {}

results = []


print("\n" + "=" * 60)
print("TRAINING 7 MODELS")
print("=" * 60)


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )


    model.fit(
        X_model_train_processed,
        y_model_train
    )


    trained_models[name] = model


    if hasattr(
        model,
        "predict_proba"
    ):

        validation_probabilities = (
            model.predict_proba(
                X_validation_processed
            )[:, 1]
        )

    else:

        validation_probabilities = (
            model.predict(
                X_validation_processed
            )
        )


    threshold = find_best_threshold(
        y_validation,
        validation_probabilities
    )


    validation_predictions = (
        validation_probabilities >=
        threshold
    ).astype(int)


    validation_accuracy = (
        accuracy_score(
            y_validation,
            validation_predictions
        )
    )


    validation_precision = (
        precision_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )
    )


    validation_recall = (
        recall_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )
    )


    validation_f1 = (
        f1_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )
    )


    try:

        validation_roc_auc = (
            roc_auc_score(
                y_validation,
                validation_probabilities
            )
        )

    except Exception:

        validation_roc_auc = np.nan


    try:

        validation_pr_auc = (
            average_precision_score(
                y_validation,
                validation_probabilities
            )
        )

    except Exception:

        validation_pr_auc = np.nan


    results.append({

        "Model": name,

        "Threshold": threshold,

        "Accuracy": validation_accuracy,

        "Precision": validation_precision,

        "Recall": validation_recall,

        "F1": validation_f1,

        "ROC_AUC": validation_roc_auc,

        "PR_AUC": validation_pr_auc

    })


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


results_df = results_df.sort_values(
    by=[
        "PR_AUC",
        "F1",
        "Recall"
    ],
    ascending=False
)


print("\n" + "=" * 60)
print("VALIDATION MODEL COMPARISON")
print("=" * 60)


print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 16. SELECT BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)


best_threshold = float(
    results_df.iloc[0]["Threshold"]
)


best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 60)
print("SELECTED MODEL")
print("=" * 60)


print(
    "Selected model:",
    best_model_name
)


print(
    "Selected threshold:",
    best_threshold
)


# ============================================================
# 17. FINAL TEST EVALUATION
# ============================================================

test_probabilities = (
    best_model.predict_proba(
        X_test_processed
    )[:, 1]
)


test_predictions = (
    test_probabilities >=
    best_threshold
).astype(int)


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)


test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0
)


test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0
)


test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0
)


test_roc_auc = roc_auc_score(
    y_test,
    test_probabilities
)


test_pr_auc = average_precision_score(
    y_test,
    test_probabilities
)


print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)


print(
    f"Accuracy : {test_accuracy:.4f}"
)

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall   : {test_recall:.4f}"
)

print(
    f"F1       : {test_f1:.4f}"
)

print(
    f"ROC-AUC  : {test_roc_auc:.4f}"
)

print(
    f"PR-AUC   : {test_pr_auc:.4f}"
)


print("\nClassification Report:")


print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "Low Risk",
            "High Risk"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix:")


print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)


# ============================================================
# 18. SAVE MODEL ARTIFACTS
# ============================================================

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "risk_agent"
    / "models"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)


joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl"
)


with open(
    MODEL_DIR / "feature_columns.json",
    "w"
) as f:

    json.dump(
        list(ML_FEATURES),
        f,
        indent=4
    )


# Save threshold
with open(
    MODEL_DIR / "risk_threshold.json",
    "w"
) as f:

    json.dump(
        {
            "threshold": best_threshold
        },
        f,
        indent=4
    )


# ============================================================
# 19. VERIFY FILES
# ============================================================

print("\n" + "=" * 60)
print("MODEL ARTIFACTS SAVED")
print("=" * 60)


print(
    "Selected model:",
    type(best_model).__name__
)


print(
    "Threshold:",
    best_threshold
)


print(
    "Model:",
    MODEL_DIR / "best_model.pkl"
)


print(
    "Preprocessor:",
    MODEL_DIR / "preprocessor.pkl"
)


print(
    "Features:",
    MODEL_DIR / "feature_columns.json"
)


print(
    "Threshold:",
    MODEL_DIR / "risk_threshold.json"
)


print("\n" + "=" * 60)
print("VERIFYING SAVED FILES")
print("=" * 60)


for filename in [

    "best_model.pkl",

    "preprocessor.pkl",

    "feature_columns.json",

    "risk_threshold.json"

]:

    filepath = (
        MODEL_DIR / filename
    )


    print(
        filename,
        "->",
        "FOUND"
        if filepath.exists()
        else "MISSING"
    )


print(
    "\nTraining completed successfully."
)