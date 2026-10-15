# TensorFlow 工具書：從資料到訓練、驗證與部署

[← 首頁](../README.md) · [TensorFlow ↔ PyTorch 對照](framework-compare.md) · [概念速查](dl-concepts.md) · [3 本練習 notebook](../modern/tensorflow/README.md) · [深度學習流程圖](diagrams/dl-framework-flow.html) · [來源筆記對照與勘誤](note-map.md) · [可執行分類範例](../examples/tf_classification.py)

本章整理另附的 `ML&DL_筆記.md` 與 repo 內 Day 63–100 的主題，採用現行 **TensorFlow 2 + `tf.keras`（Keras 3）** API。Keras 仍是 TensorFlow 的高階模型介面，適合標準訓練與回呼；需要逐步控制梯度時用 `tf.GradientTape`。本書另保留 [PyTorch 路線](pytorch.md)，兩者可依專案環境選擇。舊筆記的 Python 3.8、TensorFlow 2.4／2.5、CUDA 手動複製設定只用於重現當年環境。

## 1. 何時使用哪一層 API

| 你需要做什麼 | TensorFlow 入口 | 先判斷 |
|---|---|---|
| 小型表格分類／回歸 | 先比 [scikit-learn 基準](ml.md) | 神經網路的額外成本是否有實際收益 |
| 固定順序、單一路徑模型 | `tf.keras.Sequential`、`Dense`／`Conv2D` | 適合先跑通資料、loss 與驗證 |
| 多輸入、殘差或多輸出 | `tf.keras.Input` + Functional `Model` | 分支與合併在模型圖上清楚可見 |
| 動態控制流或特殊層 | 子類化 `tf.keras.Model`，必要時覆寫 `call` | 先確認 Functional API 是否已足夠 |
| 一般訓練、早停、最佳模型 | `compile` → `fit` + `callbacks` | 驗證資料獨立於訓練資料 |
| 自訂更新步驟／梯度 | `tf.GradientTape` + `optimizer.apply_gradients` | loss 必須保持張量計算路徑 |
| 大量或串流資料 | `tf.data.Dataset` | `fit(dataset)` 可以逐批讀取；是否佔滿記憶體取決於資料來源與快取 |
| 推論服務 | `model.export(...)` 產生 SavedModel | 訓練續跑用 `.keras`，服務部署用匯出產物 |

### 安裝與裝置

在**獨立虛擬環境**中安裝；本 repo 的 [`requirements-tensorflow.txt`](../requirements-tensorflow.txt) 用於 CPU 教學。Python／GPU 相容性請以 [TensorFlow 官方安裝頁](https://www.tensorflow.org/install/pip) 當下的列表為準。

```bash
python -m venv .venv-tf
# 啟用環境後
python -m pip install -r requirements-tensorflow.txt
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
python examples/tf_classification.py
```

原筆記的原生 Windows CUDA 步驟不適用於現代 TensorFlow GPU：官方文件指出 **2.10 是最後支援原生 Windows GPU 的版本**；新版本若要 NVIDIA GPU，走 Linux／WSL2 的官方路徑。CPU 輸出 `[]` 是正常狀態。不要依原筆記自行把 cuDNN 複製到 CUDA 目錄，也不要用已棄用的 `tf.test.is_gpu_available()` 作安裝判斷。[安裝說明](https://www.tensorflow.org/install/pip)

## 2. 最小分類流程：資料 → 模型 → loss → fit

先按資料產生方式切分 train／validation／test，再只用 train 擬合任何標準化或詞彙表。若同一人／裝置有多筆資料，先按群組切；時間資料按時間切。下面用不需下載的合成資料示範 API；真實任務仍須保留最後一次測試集。

```python
import tensorflow as tf

# x_train: float32 [N, 2]；y_train: 整數類別 [N]
train = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
         .shuffle(len(x_train), seed=42).batch(64).prefetch(tf.data.AUTOTUNE))
valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(64)

model = tf.keras.Sequential([
    tf.keras.Input(shape=(2,)),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(2),  # 每類一個 logit
])
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=0.01, weight_decay=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)
history = model.fit(train, validation_data=valid, epochs=20)
```

這段的完整資料生成、早停與保存見 [可執行版本](../examples/tf_classification.py)。`tf.data` 以 `from_tensor_slices` 包住已在記憶體中的 NumPy 陣列；若資料很大，應由檔案建立 dataset 或串流讀取，而非先讀成整個陣列。`shuffle` 僅用於訓練資料；驗證與測試保持穩定順序。[tf.data 指南](https://www.tensorflow.org/guide/data)

### loss、標籤、輸出要配對

| 任務 | 最後一層 | 標籤 | loss | 推論時轉機率 |
|---|---|---|---|---|
| 二元分類 | `Dense(1)` | `float32`，形狀 `[N, 1]` | `BinaryCrossentropy(from_logits=True)` | `tf.sigmoid(logits)` |
| 多類互斥、類別索引 | `Dense(C)` | 整數 `[N]` | `SparseCategoricalCrossentropy(from_logits=True)` | `tf.nn.softmax(logits, axis=-1)` |
| 多類互斥、one-hot | `Dense(C)` | float `[N, C]` | `CategoricalCrossentropy(from_logits=True)` | `tf.nn.softmax` |
| 多標籤可同時為真 | `Dense(C)` | float `[N, C]` | `BinaryCrossentropy(from_logits=True)` | 每類 `tf.sigmoid` |
| 連續數值回歸 | `Dense(1)` | float `[N, 1]` | `MeanAbsoluteError` 或 `MeanSquaredError` | 不必轉機率 |

採用 `from_logits=True` 時，最後一層不先加 sigmoid／softmax。指標仍需依任務選擇：不平衡分類看 precision、recall、PR-AUC 與業務代價；回歸看 MAE／RMSE。Softmax 將**任意實數 logits**轉成總和為 1 的機率，負 logits 並不代表「不能算機率」。

## 3. 想掌控每個 batch：GradientTape

`fit` 已可處理多數標準訓練；需自訂多個 loss 或更新規則時才寫迴圈。`GradientTape` 預設追蹤被使用的可訓練變數；一般輸入張量若需對它求導，則要明確 `tape.watch(x)`。[官方訓練迴圈](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
for xb, yb in train:
    with tf.GradientTape() as tape:
        logits = model(xb, training=True)
        loss = loss_fn(yb, logits)
        if model.losses:  # 層上另設 kernel_regularizer 時
            loss += tf.add_n(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

驗證時呼叫 `model(xb, training=False)`，計算完整驗證集指標，不執行 `apply_gradients`。若自訂 loss 中途把張量轉成 `.numpy()` 再交給 Python 計算，梯度路徑會斷開。`tf.Variable` 是可訓練參數的一種；普通常數亦可透過 `tape.watch` 求導，原筆記「沒有設 Variable 就一定不能微分」過於絕對。

## 4. 過擬合、回呼與儲存

| 症狀／目的 | API | 使用判斷 |
|---|---|---|
| train 好、validation 差 | `Dropout`、`kernel_regularizer`、資料增強 | 先排除資料洩漏與切分錯誤 |
| 過多 epoch | `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)` | 監控驗證集；`min_delta` 是改善門檻 |
| 要保留最佳模型 | `ModelCheckpoint("best.keras", monitor="val_loss", save_best_only=True)` | 檔名副檔名符合 Keras 3 要求 |
| 驗證停滯要降學習率 | `ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)` | 先檢查初始學習率與資料尺度 |
| 推論結果有異 | `model(x, training=False)` | 關閉 Dropout，BatchNorm 使用推論統計 |

**繼續訓練**：`model.save("model.keras")`，再用 `tf.keras.models.load_model("model.keras")`。`compile=False` 的意思是載入後**不編譯**，不是「沿用原本 optimizer／loss」；若要繼續 `fit`，需重新 `compile`。**只存權重**：`model.save_weights("model.weights.h5")`，重建相容架構後 `load_weights`。**部署推論**：`model.export("serving_dir")` 產生 SavedModel，可供 TensorFlow Serving 使用；Keras 3 的 `load_model` 不用來直接載入此匯出目錄。[Keras 保存](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/) · [匯出](https://keras.io/api/models/model_saving_apis/export/) · [ModelCheckpoint](https://keras.io/api/callbacks/model_checkpoint/)

## 5. 影像：CNN、增強與預訓練

TensorFlow 預設影像形狀 `[N, H, W, C]`，`Conv2D` 核心形狀 `[Hk, Wk, Cin, Cout]`；每個輸出通道會跨全部輸入通道學習權重。`padding="same"` **允許 stride > 1**，此時輸出仍可能縮小；它不表示任何 stride 都保持原尺寸。池化通常不含可學參數，但會改變高寬。GAP 可減少進入分類頭的維度，不保證每個任務的準確率都更高或更低。

```python
cnn = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(n_classes),
])
```

訓練集可用 `tf.keras.layers.RandomFlip("horizontal")`、`RandomRotation(...)` 等增強，前提是變換不改標籤；驗證／測試只做固定前處理。`fit` 能直接吃 `tf.data.Dataset`；舊 `fit_generator` 教法沒有必要，也不能推論 `fit` 必定把資料全塞進 RAM。

少量影像時先評估預訓練網路：

```python
inputs = tf.keras.Input(shape=(160, 160, 3))  # 原始 0–255 像素
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
base = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3), include_top=False, weights="imagenet"
)  # 首次會下載權重；離線測試可用 weights=None，但不算遷移學習
base.trainable = False
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(n_classes)(x)
model = tf.keras.Model(inputs, outputs)
```

先只訓練新分類頭，再視驗證結果解凍後段並以小學習率重新 `compile`。含 BatchNorm 的 backbone 微調時仍建議 `base(x, training=False)`，避免更新其移動統計。[Keras 遷移學習指南](https://keras.io/guides/transfer_learning/)

## 6. 序列：Embedding、GRU、LSTM、注意力

序列張量是 `[batch, time, feature]`。文字先做詞彙表與整數化，再用 `Embedding`；數值時間序列直接以特徵維度輸入。GRU 與 LSTM 是基準選項；若需要長距離關係或預訓練文字模型，再比較注意力／Transformer。舊筆記的 RNN、VGG、GoogLeNet、ResNet 是架構史與概念材料，不代表每個新專案都要手刻這些網路。

```python
sequence_model = tf.keras.Sequential([
    tf.keras.Input(shape=(None, n_features)),
    tf.keras.layers.GRU(32),  # 最後時間步表示：[N, 32]
    tf.keras.layers.Dense(1),
])
```

疊第二個循環層時，前一層需 `return_sequences=True`，輸出才是 `[N, T, hidden]`。變長序列需 padding 與 mask；時間預測按時間切分，縮放器只在訓練期擬合，特徵僅用預測時已知的資訊。股票價格等高噪音任務先比「上一值／季節性」基準，不因用了 GRU 就能推論有預測力。[GRU API](https://keras.io/api/layers/recurrent_layers/gru/)

## 7. 排錯順序

1. **shape／dtype**：印出一批 `xb.shape`、`yb.shape`；核對 `[N,H,W,C]`、標籤與 loss 配對。
2. **資料切分**：群組／時間有無穿越邊界？標準化是否只在 train 擬合？
3. **loss 不下降**：先看 logits、學習率、是否能在很小資料上過擬合；再查梯度是否 `None`。
4. **validation 變差**：檢查增強是否只作用訓練集、前處理一致性、模型容量與資料分布。
5. **部署結果不同**：使用相同前處理、類別順序與推論模式，檢查匯出產物的輸入簽名。

相關文件：[TensorFlow/Keras 訓練](https://keras.io/guides/training_with_built_in_methods/) · [tf.data](https://www.tensorflow.org/guide/data) · [PyTorch 對照](framework-compare.md)。
