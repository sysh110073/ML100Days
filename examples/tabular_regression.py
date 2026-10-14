"""連續數值預測：相同切分下比較基準、Ridge 與樹集成。"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    rng = np.random.default_rng(42)
    values = rng.normal(size=(600, 3))
    X = pd.DataFrame(values, columns=["size", "age", "distance"])
    y = 30 + 8 * X["size"] - 3 * X["distance"] + 5 * X["age"].pow(2) + rng.normal(0, 2, 600)
    X.loc[rng.choice(len(X), 25, replace=False), "distance"] = np.nan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "dummy": make_pipeline(SimpleImputer(strategy="median"), DummyRegressor(strategy="median")),
        "ridge": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0)),
        "hist_gb": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingRegressor(random_state=42)),
    }
    for name, model in candidates.items():
        cv_mae = -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error").mean()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        print(
            f"{name}: CV MAE={cv_mae:.3f}, test MAE={mean_absolute_error(y_test, prediction):.3f}, "
            f"test RMSE={root_mean_squared_error(y_test, prediction):.3f}"
        )


if __name__ == "__main__":
    main()
