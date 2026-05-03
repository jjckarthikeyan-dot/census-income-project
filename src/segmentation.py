from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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

def select_segmentation_features(df):
    segment_features = [
        "age",
        "education",
        "class of worker",
        "marital stat",
        "major industry code",
        "major occupation code",
        "sex",
        "race",
        "full or part time employment stat",
        "capital gains",
        "capital losses",
        "dividends from stocks",
        "num persons worked for employer",
        "weeks worked in year",
        "own business or self employed",
    ]

    return df[segment_features].copy(), segment_features

def build_segmentation_pipeline(X, n_segments=5):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_features),
        ("categorical", categorical_pipe, categorical_features),
    ])

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("svd", TruncatedSVD(n_components=25, random_state=42)),
        ("cluster", MiniBatchKMeans(
            n_clusters=n_segments,
            random_state=42,
            batch_size=4096,
            n_init=10,
        )),
    ])

    return pipeline, numeric_features, categorical_features

def fit_segmentation_model(pipeline, X, sample_size=40000):
    fit_size = min(sample_size, len(X))
    fit_X = X.sample(n=fit_size, random_state=42)

    pipeline.fit(fit_X)

    return pipeline, fit_size

def score_segments(pipeline, X, sample_size=80000):
    score_size = min(sample_size, len(X))
    score_X = X.sample(n=score_size, random_state=7).copy()

    score_X["segment"] = pipeline.predict(score_X)

    return score_X, score_size

def calculate_silhouette_score(pipeline, X, sample_size=10000):
    eval_size = min(sample_size, len(X))
    eval_X = X.sample(n=eval_size, random_state=21)

    transformed = pipeline.named_steps["preprocess"].transform(eval_X)
    reduced = pipeline.named_steps["svd"].transform(transformed)
    labels = pipeline.named_steps["cluster"].predict(reduced)

    score = silhouette_score(reduced, labels)

    return float(score), eval_size

def get_top_categories(group, columns, n=3):
    output = {}

    for col in columns:
        values = (
            group[col]
            .astype(str)
            .value_counts(normalize=True)
            .head(n)
        )

        output[col] = {
            str(k): round(float(v), 3)
            for k, v in values.items()
        }

    return output

def build_segment_profile(scored_df, original_df):
    profile_df = original_df.loc[scored_df.index].copy()
    profile_df["segment"] = scored_df["segment"]

    chosen_cat_cols = [
        "education",
        "class of worker",
        "marital stat",
        "major industry code",
        "major occupation code",
        "sex",
        "race",
    ]

    rows = []

    for segment, group in profile_df.groupby("segment"):
        rows.append({
            "segment": int(segment),
            "records": int(len(group)),
            "population_weight": round(float(group["weight"].sum()), 1),
            "income_gt_50k_rate": round(float(group["target"].mean()), 4),
            "avg_age": round(float(group["age"].mean()), 1),
            "avg_weeks_worked": round(float(group["weeks worked in year"].mean()), 1),
            "avg_capital_gains": round(float(group["capital gains"].mean()), 1),
            "avg_capital_losses": round(float(group["capital losses"].mean()), 1),
            "avg_dividends": round(float(group["dividends from stocks"].mean()), 1),
            "top_categories": get_top_categories(group, chosen_cat_cols, n=3),
        })

    return pd.DataFrame(rows).sort_values("segment")

def add_business_labels(profile):
    labels = []

    for _, row in profile.iterrows():
        income_rate = row["income_gt_50k_rate"]
        avg_age = row["avg_age"]
        weeks = row["avg_weeks_worked"]

        if income_rate >= 0.30:
            label = "High-income professional segment"
            recommendation = "Use premium product offers, investment-oriented messaging, and loyalty upgrades."
        elif weeks >= 35 and income_rate >= 0.12:
            label = "Stable working middle-income segment"
            recommendation = "Use value bundles, practical promotions, and family-focused offers."
        elif avg_age < 30:
            label = "Younger early-career segment"
            recommendation = "Use entry-level pricing, digital campaigns, and education/career-related offers."
        elif weeks < 20:
            label = "Low labor-force attachment segment"
            recommendation = "Use budget-focused promotions and avoid high-price premium campaigns."
        else:
            label = "General value-seeking segment"
            recommendation = "Use broad discount campaigns and need-based product recommendations."

        labels.append({
            "segment": int(row["segment"]),
            "business_label": label,
            "marketing_recommendation": recommendation,
        })

    label_df = pd.DataFrame(labels)

    return profile.merge(label_df, on="segment", how="left")

def save_outputs(pipeline, profile, scored_df, original_df, segment_features, metadata):
    profile.to_csv(OUT / "segment_profile.csv", index=False)

    export_df = original_df.loc[scored_df.index].copy()
    export_df["segment"] = scored_df["segment"]

    export_cols = ["segment", "label", "target", "weight"] + segment_features

    export_df[export_cols].to_csv(
        OUT / "census_with_segments_sample.csv",
        index=False,
    )

    with open(OUT / "segmentation_summary.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    joblib.dump(pipeline, OUT / "segmentation_pipeline.joblib")

def main():
    df = load_data()

    X, segment_features = select_segmentation_features(df)

    pipeline, numeric_features, categorical_features = build_segmentation_pipeline(
        X,
        n_segments=5,
    )

    pipeline, fit_size = fit_segmentation_model(
        pipeline,
        X,
        sample_size=40000,
    )

    scored_df, score_size = score_segments(
        pipeline,
        X,
        sample_size=80000,
    )

    silhouette, silhouette_sample_size = calculate_silhouette_score(
        pipeline,
        X,
        sample_size=10000,
    )

    profile = build_segment_profile(scored_df, df)
    profile = add_business_labels(profile)

    metadata = {
        "model": "MiniBatchKMeans with OneHotEncoder and TruncatedSVD",
        "n_segments": 5,
        "fit_sample_size": fit_size,
        "profile_sample_size": score_size,
        "silhouette_sample_size": silhouette_sample_size,
        "silhouette_score": round(silhouette, 4),
        "inertia": round(float(pipeline.named_steps["cluster"].inertia_), 2),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "business_note": (
            "Segments were created using demographic and employment features only. "
            "Income label and observation weight were not used as clustering inputs."
        ),
    }

    save_outputs(
        pipeline,
        profile,
        scored_df,
        df,
        segment_features,
        metadata,
    )

    print("\nCustomer Segmentation Profile")
    print(profile.to_string(index=False))

    print("\nSegmentation files saved:")
    print("- outputs/segment_profile.csv")
    print("- outputs/census_with_segments_sample.csv")
    print("- outputs/segmentation_summary.json")
    print("- outputs/segmentation_pipeline.joblib")


if __name__ == "__main__":
    main()