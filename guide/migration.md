# 舊作業 → 現代 API 與方法遷移

[← 工具書首頁](../README.md) · [舊作業索引](legacy-index.md)

更新檢視：2026-10-14。原 `homework/` notebook 保留當年的提問與結果，用來對照學習歷程；請使用 `guide/` 和 `examples/` 的語法執行新練習。原作業的 Kaggle CSV、外部圖片、模型檔與套件環境未存入 repo，故不能把「原檔全部可重跑」當成驗收條件。

## 直接會壞或明顯過時的寫法

| 舊寫法／出現範圍 | 現代寫法 | 為什麼 |
|---|---|---|
| `DataFrame.append(other)`：Day 5、7、10、12、22–32 等 | `pd.concat([left, right], ignore_index=True)`；逐列累積先建 list，最後一次轉 DataFrame | pandas 已移除 `append()`；反覆 concat 也低效 |
| `np.int`、`np.float`：Day 55、57–59、74、91 | `int`、`float` 或明確 `np.int64`、`np.float64` | NumPy 已讓這些舊別名失效 |
| `sns.distplot(...)`：Day 16、19、23 | `sns.histplot(data=df, x="column", kde=True)`；密度獨立用 `sns.kdeplot` | `distplot` 已棄用 |
| `datasets.load_boston()`：Day 38、40、42、44 | 離線練習用 `load_diabetes()`；房價主題可用 `fetch_california_housing()`（需下載） | Boston 資料集已移除，也有倫理問題 |
| `OneHotEncoder(sparse=False)` 或手動固定欄位 | `OneHotEncoder(sparse_output=False, handle_unknown="ignore")`；建模時放 `ColumnTransformer` | 參數已改名；新類別要可處理 |
| 在全資料上 `fillna`／`StandardScaler.fit` 後切分 | 先切分，再用 `Pipeline` / `ColumnTransformer` | 避免驗證／測試資料洩漏 |
| 將類別用 `LabelEncoder` 轉數字後當一般連續特徵 | 名目類別用 `OneHotEncoder`，序位類別明確指定 `OrdinalEncoder` | 數字大小可能被模型誤解 |
| 手動以全訓練資料目標平均編碼同一批訓練列（Day 25） | `TargetEncoder` 的 `fit_transform` 交叉擬合並包進 CV | 同一列目標回流造成洩漏與過擬合 |
| 單次隨機切分用於重複客戶／時間資料 | `GroupKFold`／`TimeSeriesSplit` 或時間留後測試 | 相依紀錄進入兩邊會高估表現 |
| `mlxtend.classifier.StackingClassifier`（Day 51–53） | 優先用 `sklearn.ensemble.StackingClassifier`／`StackingRegressor` | 減少額外依賴；CV 產生次層輸入 |

以上是**遷移指引**，不是機械式全域取代。例如 `append` 也可能是 Python list 的合法方法，不能看到字串就全部換掉。`load_diabetes()` 與 California 房價資料的目標、欄位、規模不同，無法拿原 notebook 的成績直接比較。

## 方法上的修正

- **Day 34「把測試集切成平衡類別」**：模型評估的測試分布應盡量代表部署時的分布。若只為比較類別別表現，可另外做平衡子集分析，但不能只報該分數。
- **Day 54「無監督學習不能評估」**：沒有標籤仍可看輪廓係數、群穩定性、密度、異常覆核與下游用途；內部指標不能單獨證明群有業務價值。
- **Day 63「CNN 只適合影像、RNN 只適合時間」**：卷積也可用於序列，注意力模型也可處理影像與時間序列。依資料結構、可用預訓練模型、成本與驗證結果選架構。
- **Day 66–100 Keras／TensorFlow 程式**：現代教學改用 [PyTorch](pytorch.md) 的 `Dataset`、`DataLoader`、`nn.Module`、明確訓練迴圈。Day 90–91 的顏色直方圖／HOG + SVM 是傳統 CV 特徵，不必硬改成神經網路；若載入 CIFAR 資料，改用 `torchvision.datasets`。

## Keras → PyTorch 對照

| 原教學概念 | PyTorch 寫法 | 注意 |
|---|---|---|
| `keras.datasets` | `torchvision.datasets`／自訂 `Dataset` | 影像資料轉換在 dataset 中定義 |
| `Sequential([...])` | `nn.Sequential(...)` | 複雜多輸入／分支用自訂 `nn.Module.forward` |
| `Dense`、`Conv2D` | `nn.Linear`、`nn.Conv2d` | PyTorch 影像 shape 是 `[N, C, H, W]` |
| `model.compile(loss, optimizer)` | 建立 `criterion`、`optimizer` | 訓練迴圈明確控制每一步 |
| `model.fit` | `for epoch` + `for batch in DataLoader` | 記得 `zero_grad` → forward → loss → backward → step |
| `model.evaluate` | `model.eval()` + `torch.inference_mode()` | 累計整個驗證集，勿只看最後 batch |
| `model.predict` | `model.eval()` + forward；需要時再 softmax／sigmoid | `CrossEntropyLoss` 前不要 softmax |
| `Dropout`、`BatchNormalization` | `nn.Dropout`、`nn.BatchNorm1d/2d` | train/eval 模式會改變其行為 |
| L2 regularizer | `AdamW(weight_decay=...)` 或合適 optimizer | 不要將 L2 與解耦 weight decay 視為所有 optimizer 下完全等價 |
| `EarlyStopping`、`ModelCheckpoint` | 驗證迴圈 + patience + `torch.save(state_dict)` | 載入最佳權重；驗證集不參與梯度 |
| `ReduceLROnPlateau` | `torch.optim.lr_scheduler.ReduceLROnPlateau` | 驗證後 `scheduler.step(val_loss)` |
| `ImageDataGenerator` | `torchvision.transforms.v2` | 隨機增強只作用於訓練集 |
| `ResNet50(weights="imagenet")` | `resnet18(weights=ResNet18_Weights.DEFAULT)` 等 | 配合 `weights.transforms()`；首次下載權重 |
| 儲存完整 Keras 模型 | `torch.save(model.state_dict(), path)` | 另存模型設定／類別對照；載入 `weights_only=True` |

## 版本選擇與更新方式

工具書以目前文件仍支援的 API 為準，`requirements-modern.txt` 給出保守的版本範圍；PyTorch／torchvision 依 [官方平台安裝選擇器](https://pytorch.org/get-started/locally/) 配對。要重現特定實驗時，另鎖定實際安裝版本與資料版本；不要將「最新」視為會永遠固定的版本號。

官方來源：[pandas 2.0 release notes](https://pandas.pydata.org/docs/whatsnew/v2.0.0.html)、[NumPy 1.24 release notes](https://numpy.org/doc/1.24/release/1.24.0-notes.html)、[seaborn distplot 棄用說明](https://seaborn.pydata.org/generated/seaborn.distplot.html)、[OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)、[TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)、[PyTorch Tutorials](https://docs.pytorch.org/tutorials/)。
