# 有標籤資料：先選評估方式，再選模型

[← 工具書首頁](../README.md) · [模型流程圖](diagrams/supervised-flow.html) · [分類範例](../examples/tabular_classification.py) · [回歸範例](../examples/tabular_regression.py)

## 1. 定義任務與評估

| 問題 | 判斷 | 第一個實作 |
|---|---|---|
| 要預測一個離散類別？ | 二元、多類或多標籤；每類樣本數 | 分類：`DummyClassifier` → `LogisticRegression` |
| 要預測連續數值？ | 單位、可接受誤差、極端值代價 | 回歸：`DummyRegressor` → `Ridge` |
| 同一人／設備重複觀測？ | 訓練與驗證不能共享實體 | `GroupKFold`；分類且不平衡可試 `StratifiedGroupKFold` |
| 依時間預測未來？ | 特徵是否在預測時已知 | 最後一段時間留作測試；開發時 `TimeSeriesSplit`，必要時留 gap |
| 獨立樣本、類別不均衡？ | 每類在各切分是否足夠 | `train_test_split(..., stratify=y)`；交叉驗證用 `StratifiedKFold` |

**測試集只用於最後一次評估**。模型、超參數、缺值填補、編碼、縮放、特徵選擇與閾值都在訓練／驗證流程中決定。若特徵包含目標發生後才知道的資訊，即使使用 Pipeline 仍會洩漏。

```python
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score

# 同一客戶多列：groups 與 X/y 逐列對齊。
group_scores = cross_val_score(model, X, y, groups=customer_id,
                               cv=GroupKFold(n_splits=5), scoring="roc_auc")
# 時序資料先按時間排序；每折用過去預測未來。
time_scores = cross_val_score(model, X_sorted, y_sorted,
                              cv=TimeSeriesSplit(n_splits=5), scoring="neg_mean_absolute_error")
```

此段是兩種任務的**替代範例**：`roc_auc` 用於二元分類，`neg_mean_absolute_error` 用於回歸；不要直接把同一個分類模型套入第二行。

## 2. 分類模型怎麼挑

| 情境 | 候選模型（scikit-learn） | 為什麼先試／何時避開 | 常看指標 |
|---|---|---|---|
| 必須有基準 | `DummyClassifier` | 看模型是否真的比猜最常見類別或類別比例好 | 與正式模型同指標 |
| 表格、小到中型、要解釋性 | `LogisticRegression` | 快、可看係數；需處理缺值、類別與尺度 | ROC-AUC、PR-AUC、log loss、F1 |
| 高維稀疏文字 | `TfidfVectorizer` + `LogisticRegression`／`LinearSVC` | 線性法常是強基準；`LinearSVC` 本身不提供機率 | F1、PR-AUC；機率需求另校準 |
| 非線性表格、想看交互作用 | `RandomForestClassifier`、`HistGradientBoostingClassifier` | 樹集成處理非線性；較難像係數一樣解釋，需調葉數／深度 | ROC-AUC、PR-AUC、校準 |
| 少量資料、分界複雜 | `SVC`（RBF） | 對縮放敏感；大資料／調參成本高 | F1、ROC-AUC |
| 距離有意義且資料不大 | `KNeighborsClassifier` | 推論較慢；高維資料常不理想 | F1、accuracy |
| 計數型文字快速起點 | `MultinomialNB` | 假設較強但便宜；需非負特徵 | F1、log loss |
| 影像等非結構化資料 | 先看可用預訓練模型；自訓用 PyTorch | 標註量、計算成本與部署要求決定是否值得 | 視任務而定 |

多標籤分類（一筆可有多個標籤）可用 `OneVsRestClassifier(LogisticRegression(...))` 作基準，或在 PyTorch 用多個 logits + `BCEWithLogitsLoss`。它與互斥多類分類的 `CrossEntropyLoss` 不同。類別極度不均衡時 accuracy 往往誤導；同時看類別支持數、混淆矩陣、PR-AUC、recall 和精確率。若決策依賴概率，另看校準曲線／`CalibratedClassifierCV`。

## 3. 回歸模型怎麼挑

| 情境 | 候選模型 | 主要取捨 | 指標 |
|---|---|---|---|
| 基準 | `DummyRegressor(strategy="median")` | 衡量是否勝過固定預測 | MAE、RMSE |
| 近似線性、要解釋 | `LinearRegression`、`Ridge` | Ridge 對共線性較穩；縮放放 Pipeline | MAE、RMSE、R² |
| 希望部分係數為零 | `Lasso`、`ElasticNet` | 需調正則化強度；強相關特徵可能不穩定 | MAE、RMSE |
| 非線性表格 | `RandomForestRegressor`、`HistGradientBoostingRegressor` | 捕捉交互作用；留意外推能力 | MAE、RMSE |
| 極端誤差成本高 | `HuberRegressor` 或適當分位數模型 | 先確認業務損失；不能只換模型掩蓋資料錯誤 | 對應業務損失 |

MAE 以原單位可解讀；RMSE 對大誤差更敏感。MAPE 在真值為零或接近零時不適合。時間序列先用「上一期／季節同一期」基準，再評估其他模型；不要隨機打散時間。

## 4. 可直接套用的混合欄位分類流程

以下只示範核心結構；完整、可執行且有基準比較的版本見 [範例](../examples/tabular_classification.py)。

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric = ["age", "income"]
categorical = ["region", "channel"]
preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore")), categorical),
])
model = make_pipeline(preprocess, LogisticRegression(max_iter=1000))
# 先正確切分 X_train, X_test, y_train, y_test；再 fit。
model.fit(X_train, y_train)
probability = model.predict_proba(X_test)[:, 1]
```

這樣在交叉驗證每一折中，填補值、平均值、標準差與類別詞彙都只從該折訓練資料學到。可用 `GridSearchCV(model, {"logisticregression__C": [0.1, 1, 10]}, cv=...)` 調參；大搜尋先用 `RandomizedSearchCV`。比較候選模型時，使用相同資料切分、前處理與指標。

## 5. 指標與決策閾值

| 目標／成本 | 首選 | 何時補看其他指標 |
|---|---|---|
| 類別接近平衡，兩種錯誤代價近似 | accuracy、macro F1 | 類別別結果與混淆矩陣 |
| 少數正例、重視找到它 | recall、PR-AUC（average precision） | 精確率與假陽性成本 |
| 假陽性很貴 | precision、特定 precision 下的 recall | 漏掉正例的成本 |
| 需要比較排序能力 | ROC-AUC；不平衡時同看 PR-AUC | 排序好不代表機率已校準 |
| 使用機率做決策 | log loss、Brier score、校準曲線 | 依業務成本在驗證集定閾值 |
| 回歸平均誤差 | MAE | 大誤差很貴時加 RMSE |

二元分類的 `predict()` 預設閾值通常是 0.5；這不是業務最佳閾值。要在驗證集根據成本或 precision/recall 約束選閾值，最後才在測試集評估一次。若要比較模型不確定性，可記錄多折分數或 bootstrap 信賴區間；小樣本差距不要過度解讀。

## 6. 常見失敗模式

- **過擬合**：訓練好、驗證差。檢查洩漏、資料切分、模型複雜度與樣本量；可減小樹深、加正則化、增加資料。
- **欠擬合**：訓練與驗證都差。先查標籤品質、特徵是否有資訊、基準是否合理，再加複雜度。
- **類別不均衡**：先調指標與閾值；`class_weight="balanced"` 可作候選，但不保證有益。重採樣只在訓練折內進行。
- **看似高分的資料洩漏**：同一實體跨訓練／測試、先對全資料填補／編碼、目標時間後的欄位、使用測試集調參。先修驗證方式再談模型。
- **錯誤解釋特徵重要性**：樹內建重要性可能偏向高基數；可在未參與擬合的資料上使用 `permutation_importance`，但相關特徵仍需謹慎解讀。

參考：[scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html)、[交叉驗證](https://scikit-learn.org/stable/modules/cross_validation.html)、[模型評估](https://scikit-learn.org/stable/modules/model_evaluation.html)、[監督式學習](https://scikit-learn.org/stable/supervised_learning.html)。
