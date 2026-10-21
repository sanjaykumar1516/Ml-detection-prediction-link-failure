
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

from app.main import generate_metrics, DEFAULT_THRESHOLDS

MODEL_DIR = ROOT / "ml" / "models"
DATA_DIR = ROOT / "data"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

FEATURES = [
    "latency",
    "packet_loss",
    "jitter",
    "snr"
]

SCENARIOS = [
    "normal",
    "degradation",
    "high_latency",
    "packet_loss",
    "interference",
    "recovery",
    "handover"
]


# ----------------------------------
# GENERATE REPRODUCIBLE DATASET
# ----------------------------------

records = []

for session in range(120):
    scenario = SCENARIOS[session % len(SCENARIOS)]

    previous = None

    for step in range(35):
        sample = generate_metrics(
            scenario,
            seed=session * 1000,
            step=step,
            thresholds=DEFAULT_THRESHOLDS
        )

        row = {
            "session": session,
            "step": step,
            "scenario": scenario,
            "latency": sample["latency"],
            "packet_loss": sample["packet_loss"],
            "jitter": sample["jitter"],
            "snr": sample["snr"],
            "interference_label": int(
                scenario == "interference"
            )
        }

        records.append(row)


df = pd.DataFrame(records)

# Create next-step target within each session.
df["next_latency"] = (
    df.groupby("session")["latency"].shift(-1)
)

df = df.dropna().reset_index(drop=True)

csv_path = DATA_DIR / "synthetic_experiments.csv"

df.to_csv(csv_path, index=False)

print("Dataset saved:", csv_path)
print("Total samples:", len(df))
print("Class distribution:")
print(df["interference_label"].value_counts())


# ----------------------------------
# GROUP-BASED TRAIN / TEST SPLIT
# ----------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df,
        groups=df["session"]
    )
)

train = df.iloc[train_idx]
test = df.iloc[test_idx]

X_train = train[FEATURES]
X_test = test[FEATURES]


# ----------------------------------
# MODEL 1: CLASSIFICATION
# ----------------------------------

classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    class_weight="balanced"
)

classifier.fit(
    X_train,
    train["interference_label"]
)

y_true = test["interference_label"]
y_pred = classifier.predict(X_test)

classification_results = {
    "accuracy": float(
        accuracy_score(y_true, y_pred)
    ),
    "precision": float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_true, y_pred, zero_division=0)
    ),
    "f1": float(
        f1_score(y_true, y_pred, zero_division=0)
    ),
    "confusion_matrix": confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).tolist(),
    "test_class_balance": {
        str(k): int(v)
        for k, v in y_true.value_counts().items()
    }
}

joblib.dump(
    classifier,
    MODEL_DIR / "jamming.joblib"
)


# ----------------------------------
# MODEL 2: LATENCY REGRESSION
# ----------------------------------

regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    random_state=42
)

regressor.fit(
    X_train,
    train["next_latency"]
)

actual = test["next_latency"]
predicted = regressor.predict(X_test)

# Simple baseline:
# assume next latency equals current latency.

baseline = test["latency"]

regression_results = {
    "model_mae": float(
        mean_absolute_error(actual, predicted)
    ),
    "model_rmse": float(
        np.sqrt(mean_squared_error(actual, predicted))
    ),
    "baseline_mae": float(
        mean_absolute_error(actual, baseline)
    ),
    "baseline_rmse": float(
        np.sqrt(mean_squared_error(actual, baseline))
    )
}

joblib.dump(
    regressor,
    MODEL_DIR / "link.joblib"
)


# ----------------------------------
# SAVE EVALUATION
# ----------------------------------

evaluation = {
    "dataset": "Reproducible synthetic simulation",
    "samples": len(df),
    "split": "GroupShuffleSplit by simulation session",
    "classification": classification_results,
    "regression": regression_results,
    "limitations": [
        "Synthetic training data",
        "Not validated on real GNSS interference recordings",
        "Not validated on real satellite-network telemetry",
        "Confidence is not independently calibrated"
    ]
}

evaluation_path = MODEL_DIR / "evaluation.json"

evaluation_path.write_text(
    json.dumps(evaluation, indent=4)
)

print("\nCLASSIFICATION RESULTS")
print(json.dumps(classification_results, indent=4))

print("\nREGRESSION RESULTS")
print(json.dumps(regression_results, indent=4))

print("\nTraining completed successfully.")
