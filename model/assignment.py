from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")


def load_dataset() -> pd.DataFrame:
    """Download and load the UCI Bank Marketing full dataset."""
    with urlopen(DATASET_URL, timeout=60) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        with archive.open("bank-full.csv") as data_file:
            return pd.read_csv(data_file, sep=";")


def make_preprocessor(
    categorical_features: list[str],
    numeric_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_models(
    categorical_features: list[str], numeric_features: list[str]
) -> dict[str, Pipeline]:
    classifiers = {
        "Logistic Regression": (
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                class_weight="balanced",
                max_depth=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "K-Nearest Neighbors": (KNeighborsClassifier(n_neighbors=7, weights="distance"), True),
        "Gaussian Naive Bayes": (GaussianNB(), True),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
    }

    return {
        name: Pipeline(
            [
                (
                    "preprocessor",
                    make_preprocessor(
                        categorical_features, numeric_features, scale_numeric
                    ),
                ),
                ("classifier", classifier),
            ]
        )
        for name, (classifier, scale_numeric) in classifiers.items()
    }


def plot_metrics(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = plt.get_cmap("tab10").colors

    for axis, metric in zip(axes.flat, METRICS):
        axis.bar(results["Model"], results[metric], color=colors[: len(results)])
        axis.set_title(metric)
        axis.set_ylabel("Score")
        axis.tick_params(axis="x", labelrotation=25)
        axis.set_ylim(-1, 1) if metric == "MCC" else axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle("Classifier comparison on the stratified test set")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metric_comparison.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "metric_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_dataset()

    required_columns = {"y", "duration"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    print(f"Downloaded dataset shape: {data.shape}")
    print("\nSource feature types:")
    print(data.drop(columns="y").dtypes.to_string())

    duplicates = int(data.duplicated().sum())
    print(f"\nExact duplicate rows: {duplicates}")
    data = data.drop_duplicates().copy()

    # Treat unreported job and education as missing; retain other "unknown" categories.
    missing_value_columns = ["job", "education"]
    data[missing_value_columns] = data[missing_value_columns].replace("unknown", np.nan)
    print("\nMissing values in job and education after handling 'unknown':")
    print(data[missing_value_columns].isna().sum().to_string())

    class_counts = data["y"].value_counts().reindex(["no", "yes"], fill_value=0)
    class_percentages = data["y"].value_counts(normalize=True).reindex(
        ["no", "yes"], fill_value=0
    ) * 100
    print("\nTarget distribution:")
    print(pd.DataFrame({"count": class_counts, "percent": class_percentages}).to_string())

    # duration is post-call leakage; pdays=-1 means no earlier contact.
    data["previously_contacted"] = (data["pdays"] != -1).astype(int)
    data["pdays"] = data["pdays"].replace(-1, 0)
    X = data.drop(columns=["y", "duration"])
    y = data["y"].map({"no": 0, "yes": 1})
    if y.isna().any():
        raise ValueError("Unexpected target labels; expected only 'no' and 'yes'.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y.astype(int),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    test_data = X_test.copy()
    test_data["y"] = y_test.map({0: "no", 1: "yes"})
    test_data.to_csv(PROJECT_ROOT / "test_data.csv", index=False)

    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    print(f"\nTraining rows: {len(X_train)}; test rows: {len(X_test)}")
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    results = []
    for name, model in build_models(categorical_features, numeric_features).items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        positive_scores = model.predict_proba(X_test)[:, 1]
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": roc_auc_score(y_test, positive_scores),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "MCC": matthews_corrcoef(y_test, predictions),
            }
        )
        model_filename = name.lower().replace("-", "_").replace(" ", "_") + ".joblib"
        joblib.dump(model, MODEL_DIR / model_filename)
        print(f"Completed and saved: {name}")

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(OUTPUT_DIR / "results.csv", index=False, float_format="%.6f")
    plot_metrics(results_frame)

    print("\nTest-set results (use these measured values in your own analysis):")
    print(results_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved holdout data to {PROJECT_ROOT / 'test_data.csv'}")
    print(f"Saved metrics and chart under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
