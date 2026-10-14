"""混合數值／類別表格的無洩漏分類基準與模型比較。"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_data(n: int = 600, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 75, size=n).astype(float)
    income = rng.normal(60_000, 18_000, size=n)
    channel = rng.choice(["web", "store", "phone"], size=n, p=[0.5, 0.35, 0.15])
    logit = -3.2 + 0.035 * (age - 40) + 0.000025 * (income - 60_000) + (channel == "web") * 0.8
    probability = 1 / (1 + np.exp(-logit))
    target = pd.Series(rng.binomial(1, probability), name="target")
    frame = pd.DataFrame({"age": age, "income": income, "channel": channel})
    frame.loc[rng.choice(n, size=30, replace=False), "income"] = np.nan
    frame.loc[rng.choice(n, size=20, replace=False), "channel"] = np.nan
    return frame, target


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age", "income"]),
            (
                "cat",
                make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
                ["channel"],
            ),
        ]
    )


def main() -> None:
    X, y = make_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "dummy": Pipeline([("preprocess", make_preprocessor()), ("model", DummyClassifier(strategy="prior"))]),
        "logistic": Pipeline(
            [("preprocess", make_preprocessor()), ("model", LogisticRegression(max_iter=1000))]
        ),
    }

    for name, model in candidates.items():
        cv_ap = cross_val_score(model, X_train, y_train, cv=cv, scoring="average_precision").mean()
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        ap = average_precision_score(y_test, probability)
        auc = roc_auc_score(y_test, probability)
        print(f"{name}: CV AP={cv_ap:.3f}, test AP={ap:.3f}, test ROC-AUC={auc:.3f}")
    assert 0 < y_test.sum() < len(y_test)


if __name__ == "__main__":
    main()
