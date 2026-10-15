# 資料分析與整理：遇到什麼問題，用什麼工具？

[← 工具書首頁](../README.md) · [資料流程圖](diagrams/data-flow.html) · [可執行範例](../examples/data_workflow.py)

## 先確認資料粒度與品質

一列是一位客戶、一次交易，還是一筆客戶每日紀錄？先說清楚粒度，才知道 `merge` 是否會讓列數倍增、`groupby` 的分母是否正確。對建模資料，另外確認目標欄位、預測時間點、可在當時取得的欄位、同一實體是否重複，以及資料涵蓋的時間範圍。

```python
from pathlib import Path
import pandas as pd

path = Path("data/transactions.csv")
df = pd.read_csv(path, parse_dates=["event_time"], dtype={"customer_id": "string"})
print(df.shape, df.dtypes, df.head())
print(df.isna().sum().sort_values(ascending=False).head(20))
print(df.duplicated(subset=["transaction_id"]).sum())
print(df["event_time"].min(), df["event_time"].max())
```

ID 不應被當成數量特徵；CSV 讀取時先指定字串型別，避免前導零消失。若時間含時區，統一轉成同一時區再比較。敏感資料的輸出與圖表要先去識別。

## 情境速查

| 情境／判斷 | 優先工具與語法 | 何時改用其他方法／注意 |
|---|---|---|
| 不知道欄位與缺漏 | `df.info()`、`df.describe(include="all")`、`df.isna().mean()`、`df.nunique()` | 先看資料字典與單位；`describe()` 不會告訴你資料洩漏 |
| 只取部分列／欄 | `df.loc[mask, ["a", "b"]]`、`df.iloc[:10]` | 避免鏈式賦值；用 `.loc` 指派 |
| 數值型別讀錯 | `pd.to_numeric(..., errors="coerce")` | 轉換失敗變缺值後，要查失敗比例與原值 |
| 日期／時間 | `pd.to_datetime(..., errors="coerce", utc=True)`、`.dt`、`resample()` | 聚合前明確時區、日曆與頻率；勿把未來資訊帶回過去 |
| 缺值 | `isna()`、`fillna()`；建模用 `SimpleImputer`／`KNNImputer` | 先判斷缺值機制；插補器放在 Pipeline 中，只用訓練集 `fit` |
| 重複列 | `duplicated(subset=...)`、`drop_duplicates(...)` | 先定義唯一鍵；「完全相同」不等於「業務上重複」 |
| 離群值 | `quantile()`、箱型圖、業務界限 | 先查輸入錯誤與單位；不要只因極端就刪除；閾值需由訓練資料決定 |
| 多表合併 | `merge(..., validate="m:1", indicator=True)` | 先確認鍵唯一性與 join cardinality；查未匹配列 |
| 分組彙總 | `groupby(...).agg(...)` | 要回填到原列時用 `transform()`；避免只為簡單聚合用 `apply()` |
| 長寬表轉換 | `melt()`、`pivot_table()`、`explode()` | `pivot()` 要求鍵唯一；重複組合用 `pivot_table()` 並指定聚合函數 |
| 分布、關係、時間趨勢 | `sns.histplot`、`sns.boxplot`、`sns.scatterplot`、`sns.lineplot` | 先抽樣或聚合大資料；類別過多時圖會失真 |
| 資料超出記憶體 | `read_csv(usecols=..., dtype=..., chunksize=...)`、Parquet | 先少讀欄、縮小 dtype；仍不夠才考慮更大工具 |

## 常用語法，按問題分類

### 1. 缺值：先量測，再決定填補

```python
missing_rate = df.isna().mean().sort_values(ascending=False)
numeric_columns = df.select_dtypes(include="number").columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
```

上例只適用一般探索。若資料要進入模型，**不要先對全表取中位數**；使用 [模型篇](ml.md#4-可直接套用的混合欄位分類流程) 的 `SimpleImputer` + `Pipeline`。`KNNImputer` 可能較慢、對尺度敏感，先與中位數基準比較。缺值本身可能有訊息，可比較 `SimpleImputer(add_indicator=True)`。

### 2. 重複、合併與粒度

```python
orders = orders.drop_duplicates(subset=["order_id"], keep="last")
assert customers["customer_id"].is_unique
joined = orders.merge(
    customers[["customer_id", "segment"]],
    on="customer_id", how="left", validate="m:1", indicator=True,
)
print(joined["_merge"].value_counts(dropna=False))
assert len(joined) == len(orders)
```

若 `validate="m:1"` 報錯，先處理右表一個客戶多列的原因；不要直接刪重複以掩蓋資料粒度不一致。聚合後再 join 時，先確認鍵的組合唯一。

### 3. 數值分布、偏態與離群值

```python
import seaborn as sns

sns.histplot(data=df, x="amount", bins=30)
q1, q3 = df["amount"].quantile([0.25, 0.75])
iqr = q3 - q1
suspected = df.loc[(df["amount"] < q1 - 1.5 * iqr) |
                   (df["amount"] > q3 + 1.5 * iqr)]
```

IQR 是**檢查候選值**的經驗規則，不是刪除指令。右偏且非負的數值可試 `np.log1p`；是否改善，要在相同切分與指標下比較。像金額為負、年齡大於合理範圍，通常先查單位或資料規則。

### 4. 分組、時間與樞紐表

```python
summary = (df.groupby("segment", dropna=False)
             .agg(n=("customer_id", "size"), revenue=("amount", "sum"))
             .reset_index())
df["segment_mean"] = df.groupby("segment", dropna=False)["amount"].transform("mean")
daily = df.set_index("event_time").resample("D")["amount"].sum()
wide = df.pivot_table(index="customer_id", columns="category",
                      values="amount", aggfunc="sum", fill_value=0)
```

做預測特徵時，`segment_mean` 或歷史累積金額也必須只由預測時點之前的紀錄算出；全資料算完再切分會洩漏未來。

### 5. 類別欄位與視覺化

```python
df["tier"] = pd.Categorical(df["tier"], categories=["bronze", "silver", "gold"], ordered=True)
counts = df["tier"].value_counts(dropna=False)
sns.boxplot(data=df, x="tier", y="amount")
sns.scatterplot(data=df.sample(min(len(df), 5_000), random_state=42),
                x="age", y="amount", hue="tier")
```

類別進模型時：無序類別先用 `OneHotEncoder(handle_unknown="ignore")`；有序類別可明確指定順序再用 `OrdinalEncoder`；高基數類別可評估 `TargetEncoder`，但必須在交叉驗證內做交叉擬合。將任意類別直接編成 0/1/2 常會暗示不存在的大小關係。

## 決策前的檢查表

- 明確記錄一列代表什麼、主鍵、時間欄位、單位、目標欄位與缺值定義。
- 每個 join 記錄 `validate` 預期與合併前後列數。
- 將「資料探索」與「模型預處理」分開；後者的統計量只從訓練集學得。
- 同一個人／裝置／帳號跨多列時，用 group 切分；時間預測則按時間切分。

參考：[pandas IO](https://pandas.pydata.org/docs/user_guide/io.html)、[缺值](https://pandas.pydata.org/docs/user_guide/missing_data.html)、[合併](https://pandas.pydata.org/docs/user_guide/merging.html)、[分組](https://pandas.pydata.org/docs/user_guide/groupby.html)、[時間序列](https://pandas.pydata.org/docs/user_guide/timeseries.html)、[seaborn API](https://seaborn.pydata.org/api.html)。
