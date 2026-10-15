# TensorFlow ↔ PyTorch：同一件事怎麼寫

[← 首頁](../README.md) · [TensorFlow 詳解](tensorflow.md) · [PyTorch 詳解](pytorch.md) · [雙框架流程圖](diagrams/dl-framework-flow.html) · [來源筆記對照](note-map.md)

**讀法**：先決定資料切分、標籤、loss 和指標，再看實作語法。兩個框架能表達相同的模型概念，但沒有「函式名稱一換就完全等價」的保證。先跑同一份離線資料：[TensorFlow 範例](../examples/tf_classification.py)／[PyTorch 範例](../examples/torch_classification_pair.py)，資料來自同一個 [生成函式](../examples/paired_data.py)。架構都是 `2 → 16 → 2`，loss 都吃 logits，optimizer 都是 AdamW，切分相同；初始權重、批次順序與底層運算不同，分數無須逐位相同。

## 1. 速查：八個步驟

| 工作 | TensorFlow 2 + `tf.keras` | PyTorch | 易錯點 |
|---|---|---|---|
| 資料批次 | `tf.data.Dataset.from_tensor_slices(...).batch(64)` | `TensorDataset` + `DataLoader(batch_size=64)` | 洗牌只用 train；時間序列不隨機打散評估邊界 |
| 模型 | `tf.keras.Sequential([Dense(...), ...])` | `nn.Sequential(nn.Linear(...), ...)` | `Dense`／`Linear` 的權重儲存維度不同 |
| 多分支模型 | Functional `tf.keras.Model(inputs, outputs)` | 自訂 `nn.Module.forward` | 輸入／輸出 shape 要逐層核對 |
| 多類 loss | `SparseCategoricalCrossentropy(from_logits=True)` | `nn.CrossEntropyLoss()` | 都吃 logits 與整數類別索引 |
| 標準訓練 | `compile(...)` → `fit(...)` | 自寫 batch 迴圈 | PyTorch 記得清梯度；Keras 指標不是更新目標 |
| 自訂梯度 | `tf.GradientTape` → `apply_gradients` | `loss.backward()` → `optimizer.step()` | 轉成 Python 數值會截斷梯度 |
| 驗證／推論 | `model(x, training=False)`、`evaluate` | `model.eval()` + `torch.inference_mode()` | Dropout／BatchNorm 行為不同 |
| 保存訓練模型 | `model.save("model.keras")` | `torch.save(model.state_dict(), "model.pt")` | PyTorch 另存架構與前處理；兩種檔案不能直接互讀 |

`tf.keras` 與 PyTorch 都有可選的高低階寫法。此處用 Keras `fit` 對照 PyTorch 明確迴圈，是教學上最常見的入口；要逐步對照反向傳播，請看下一節的 `GradientTape`。

## 2. 同一個 batch 的訓練步驟

先建立對應的模型與批次資料。下列兩段都執行「前向 → loss → 求梯度 → 更新」，且都使用原始 logits。

**TensorFlow**

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3)
with tf.GradientTape() as tape:
    logits = model(xb, training=True)
    loss = loss_fn(yb, logits)
gradients = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

**PyTorch**

```python
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(xb)
loss = loss_fn(logits, yb)
loss.backward()
optimizer.step()
```

請注意兩個 loss 的參數順序不同：Keras `loss_fn(y_true, y_pred)`；PyTorch `loss_fn(prediction, target)`。Keras 層的正則化 loss 若有設定，手寫迴圈須另加 `model.losses`；`fit` 會處理。PyTorch 若使用 `AdamW(weight_decay=...)`，不可不加條件地宣稱它與 Keras 層上的 L2 regularizer 完全相同。[Keras 自訂迴圈](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch) · [PyTorch 優化教學](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)

## 3. 先看 shape，再挑 loss

| 資料／輸出 | TensorFlow | PyTorch |
|---|---|---|
| 表格 `N × F` | `[N, F]`、`float32` | `[N, F]`、`float32` |
| 影像 | `[N, H, W, C]`（NHWC，預設） | `[N, C, H, W]`（NCHW） |
| 文字／一般序列 | `[N, T, F]` | `GRU(..., batch_first=True)` 時 `[N, T, F]` |
| 多類索引標籤 | `[N]`，整數 | `[N]`，`torch.long` |
| 多類 logits | `[N, C]` + `SparseCategoricalCrossentropy(from_logits=True)` | `[N, C]` + `CrossEntropyLoss()` |
| 二元單一 logit | `[N, 1]` + `BinaryCrossentropy(from_logits=True)` | `[N, 1]` + `BCEWithLogitsLoss()`；標籤 float 同形狀 |

影像從 TensorFlow 搬到 PyTorch 時，NumPy 可用 `x.transpose(0, 3, 1, 2)`；反向用 `x.permute(0, 2, 3, 1)`。轉置後也要核對像素範圍、resize、正規化與顏色順序。不能只改維度就認定預訓練模型可以共用輸入。

## 4. 架構骨架：影像與序列

**小型 CNN**（都輸出 `C` 個 logits）：

```python
# TensorFlow: x shape [N, 32, 32, 3]
tf_cnn = tf.keras.Sequential([
    tf.keras.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(C),
])

# PyTorch: x shape [N, 3, 32, 32]
torch_cnn = torch.nn.Sequential(
    torch.nn.Conv2d(3, 16, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.AdaptiveAvgPool2d((1, 1)),
    torch.nn.Flatten(),
    torch.nn.Linear(16, C),
)
```

**GRU**（每筆序列只預測一次）：

```python
# TensorFlow: [N, T, F] → [N, 32] → [N, 1]
tf_gru = tf.keras.Sequential([
    tf.keras.Input(shape=(None, F)),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(1),
])

# PyTorch: [N, T, F] → 最後有效時間步 → [N, 1]
class TorchGRU(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.gru = torch.nn.GRU(features, 32, batch_first=True)
        self.head = torch.nn.Linear(32, 1)

    def forward(self, x):
        _, last_hidden = self.gru(x)
        return self.head(last_hidden[-1])

torch_gru = TorchGRU(F)
```

若用 padding，Keras 用 mask，PyTorch 常用 `pack_padded_sequence` 或顯式 attention mask；未處理 padding 時「最後時間步」可能是補值。時間資料的切分與推論可得特徵，優先於選 GRU 或 Transformer。[Keras GRU](https://keras.io/api/layers/recurrent_layers/gru/) · [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)

## 5. 回呼、遷移學習與部署

| 需求 | TensorFlow/Keras | PyTorch |
|---|---|---|
| 早停 | `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)` | 在每個 epoch 驗證後記錄 best loss 與 patience，恢復最佳 `state_dict` |
| 保存最佳訓練模型 | `ModelCheckpoint("best.keras", save_best_only=True)` | 驗證改善時 `torch.save(model.state_dict(), ...)` |
| 降學習率 | `ReduceLROnPlateau(monitor="val_loss")` | `torch.optim.lr_scheduler.ReduceLROnPlateau`，驗證後 `step(val_loss)` |
| 訓練增強 | `tf.keras.layers.RandomFlip` 或 `tf.data.map` | `torchvision.transforms.v2` 放在 train Dataset |
| 預訓練影像 | `tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False)` | `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` |
| 凍結 backbone | `base.trainable = False`，重新 `compile` | `param.requires_grad = False`，optimizer 只收可訓練參數 |
| 推論部署 | `model.export(...)` → SavedModel／TensorFlow Serving | 保存權重、架構及前處理，依部署環境選服務方式 |

預訓練權重首次載入一般要下載；`weights=None` 只建空架構。TensorFlow 的 `.keras` 是訓練模型保存格式；`export` 產生推論產物。PyTorch `state_dict` 是權重字典，並非直接可供 Web 服務使用的 API。**scikit-learn 也能透過一般 Web API 發布推論**，原筆記「不能 serving」不正確；所有框架都需要明確處理輸入驗證、前處理、版本與權限。[Keras export](https://keras.io/api/models/model_saving_apis/export/) · [TensorFlow Serving](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple) · [PyTorch 儲存](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)

## 6. 如何選起點

- 已有 TensorFlow 模型、需要 `fit`／callbacks 或 TensorFlow SavedModel：從 [TensorFlow 章](tensorflow.md) 開始。
- 已有 PyTorch 模型、需要逐步改訓練流程或使用 torchvision：從 [PyTorch 章](pytorch.md) 開始。
- 還沒證明需要神經網路的表格資料：先讀 [模型選擇](ml.md)，建立 Dummy、線性與樹模型基準。
- 要把舊 Keras 程式搬到 PyTorch：先比 **shape、標籤、logits、loss、切分與指標**，再換 API；不要用兩次各自隨機切分的分數推論哪個框架更好。
