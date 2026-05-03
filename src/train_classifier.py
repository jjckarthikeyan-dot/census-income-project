from pathlib import Path
import json
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "census-bureau.data"
COLS_PATH = ROOT / "data" / "census-bureau.columns"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def load_data(data_path=DATA_PATH, cols_path=COLS_PATH):
    columns = [c.strip() for c in open(cols_path, encoding="utf-8") if c.strip()]

    df = pd.read_csv(
        data_path,
        names=columns,
        na_values="?",
        skipinitialspace=True,
    )

    df["target"] = (
        df["label"].astype(str).str.strip() == "50000+."
    ).astype(int)

    return df

def prepare_features(df):
    weights = df["weight"].astype(float)
    X = df.drop(columns=["label", "target", "weight"]).copy()
    y = df["target"]
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    for col in categorical_features:
        X[col] = X[col].fillna("Missing").astype(str)

    categorical_indices = [X.columns.get_loc(col) for col in categorical_features]

    return X, y, weights, categorical_features, categorical_indices

def split_data(X, y, weights):
    return train_test_split(
        X,
        y,
        weights,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

def train_model(X_train, y_train, w_train, X_test, y_test, w_test, categorical_indices):
    train_pool = Pool(
        X_train,
        y_train,
        cat_features=categorical_indices,
        weight=w_train,
    )

    test_pool = Pool(
        X_test,
        y_test,
        cat_features=categorical_indices,
        weight=w_test,
    )

    model = CatBoostClassifier(
        iterations=700,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100,
    )

    model.fit(
        train_pool,
        eval_set=test_pool,
        early_stopping_rounds=60,
    )

    return model

def evaluate_model(model, X_test, y_test, w_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.50).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred, sample_weight=w_test)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, pred, sample_weight=w_test)
        ),
        "roc_auc": float(roc_auc_score(y_test, proba, sample_weight=w_test)),
        "average_precision_pr_auc": float(
            average_precision_score(y_test, proba, sample_weight=w_test)
        ),
        "confusion_matrix_unweighted": confusion_matrix(y_test, pred).tolist(),
        "classification_report_unweighted": classification_report(
            y_test,
            pred,
            target_names=["<=50K", ">50K"],
            output_dict=True,
        ),
    }

    return metrics

def save_outputs(model, metrics, X, y, categorical_features):
    metrics["rows"] = int(len(X))
    metrics["positive_rate_gt_50k"] = float(y.mean())
    metrics["categorical_features"] = categorical_features

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    metrics["top_15_features"] = importance.head(15).to_dict(orient="records")

    (OUT / "classification_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    importance.to_csv(OUT / "feature_importance.csv", index=False)

    model.save_model(str(OUT / "catboost_income_classifier.cbm"))
    joblib.dump(model, OUT / "income_classifier.joblib")

    return importance

def main():
    df = load_data()

    X, y, weights, categorical_features, categorical_indices = prepare_features(df)

    X_train, X_test, y_train, y_test, w_train, w_test = split_data(
        X,
        y,
        weights,
    )

    model = train_model(
        X_train,
        y_train,
        w_train,
        X_test,
        y_test,
        w_test,
        categorical_indices,
    )

    metrics = evaluate_model(model, X_test, y_test, w_test)

    importance = save_outputs(
        model,
        metrics,
        X,
        y,
        categorical_features,
    )

    print(json.dumps(metrics, indent=2))
    print("\nTop 15 important features:")
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()