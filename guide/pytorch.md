# PyTorch 工具書：何時用、怎麼訓練、怎麼排錯

[← 工具書首頁](../README.md) · [PyTorch 流程圖](diagrams/torch-flow.html) · [7 本練習 notebook](../modern/torch/README.md) · [完整可執行範例](../examples/torch_training.py) · [TensorFlow ↔ PyTorch 對照](framework-compare.md)

這章是原 Day 63–100 的現代 **PyTorch** 路線。原作業仍在 `homework/` 作為歷史紀錄；若想沿用 TensorFlow，請從 [TensorFlow 教學](tensorflow.md) 開始，再用 [同資料對照](framework-compare.md) 比較。**Keras 仍在維護，也仍是 TensorFlow 的高階教學入口。**

## 1. 先判斷是否需要深度學習

| 資料與目的 | 先用什麼 | 需要 PyTorch 時 |
|---|---|---|
| 一般表格分類／回歸 | scikit-learn 的線性模型、樹集成 | 數據量大、特殊損失／多輸入結構、神經網路有已驗證優勢 |
| 影像分類 | 預訓練 `torchvision.models` | 自訂分類頭、微調、資料增強 |
| 文字分類／生成 | 先確認可用預訓練模型 | 需微調、自訂架構或訓練流程 |
| 時序／序列 | 時間基準與特徵模型 | 長序列、複雜跨時依賴；可考慮 GRU／Transformer |
| 多標籤輸出 | 確認標籤可同時為真 | 每類獨立 logit + `BCEWithLogitsLoss` |

### 模態對架構

- **表格／固定長度向量**：`nn.Linear` + `nn.ReLU`；先與 scikit-learn 基準比較。
- **影像**：`nn.Conv2d`、`nn.MaxPool2d`、`nn.AdaptiveAvgPool2d`；小資料優先考慮預訓練 ResNet。
- **序列**：`nn.GRU`／`nn.TransformerEncoder`；要處理長度、padding mask 與時間切分。
- **多模態**：各模態分別編碼後合併；先確認每個模態在推論時都可取得。

## 2. PyTorch 最小訓練路徑

`Dataset` 定義一筆資料；`DataLoader` 打包與洗牌；`nn.Module` 定義可學參數；loss 定義錯誤；optimizer 更新參數。完整範例已包含訓練／驗證分離、早停、降學習率、最佳權重與重新載入。

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = DataLoader(TensorDataset(X_train_float32, y_train_int64),
                    batch_size=32, shuffle=True, num_workers=0)
model = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(),
                      nn.Linear(64, n_classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

model.train()
for xb, yb in loader:
    xb, yb = xb.to(device), yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = criterion(logits, yb)
    loss.backward()
    optimizer.step()
```

`CrossEntropyLoss` 接受**原始 logits**，不要先做 `softmax`；多類目標是 `torch.long` 的類別索引。二元單一輸出可用 `BCEWithLogitsLoss`，目標為 float，shape 與 logits 一致。回歸可從 `MSELoss` 或 `L1Loss` 開始。驗證時必須 `model.eval()` 並用 `torch.inference_mode()`；訓練時切回 `model.train()`，否則 Dropout／BatchNorm 行為錯誤。

```python
model.eval()
with torch.inference_mode():
    for xb, yb in validation_loader:
        logits = model(xb.to(device))
        # 在這裡累計驗證 loss 與指標；不要更新 optimizer。
```

## 3. Day 71–89 的常見選項

| 遇到的現象／目的 | PyTorch API | 判斷與注意 |
|---|---|---|
| 多類分類 | `nn.CrossEntropyLoss()` | logits shape `[N, C]`；標籤 shape `[N]`、long |
| 二元／多標籤 | `nn.BCEWithLogitsLoss()` | logits 與 float 標籤同 shape；需要不平衡權重可評估 `pos_weight` |
| 回歸 | `nn.MSELoss()`／`nn.L1Loss()` | 對照業務指標；MSE 對極端誤差更敏感 |
| 自訂 loss | 張量運算後回傳 scalar | 不要 `.detach()`、`.item()` 後再求梯度；檢查數值穩定性 |
| 訓練震盪或發散 | `AdamW`、`SGD(momentum=...)`；調小 `lr` | 先檢查資料尺度、標籤與 loss，別只盲調 optimizer |
| 過擬合 | `weight_decay`、`nn.Dropout`、資料增強、早停 | 先檢查洩漏與 train/val 分布；比較同一切分 |
| BatchNorm | `nn.BatchNorm1d`／`nn.BatchNorm2d` | 極小 batch 的統計不穩；必須正確切換 train/eval |
| 驗證指標不再改善 | 自寫 early stopping | 儲存最佳 `state_dict`，不是最後 epoch |
| 學習率需隨表現下降 | `ReduceLROnPlateau` | 每個 epoch 驗證後呼叫 `scheduler.step(val_loss)` |
| 類別別 true positive | 累計混淆矩陣 | 每 epoch 用 `torch.inference_mode()` 在驗證集計算 |

PyTorch 沒有把舊 Keras callback 名稱一比一搬入核心訓練迴圈；早停、最佳模型保存與自訂紀錄通常是訓練迴圈中的明確邏輯。完整做法見 [範例](../examples/torch_training.py)。

## 4. CNN、資料增強與遷移學習

影像張量採 `[batch, channel, height, width]`。`Conv2d` 的 `padding=1` 在 3×3、stride 1 時維持空間尺寸；`MaxPool2d(2)` 通常讓高寬減半。以下是小型 CNN 的骨架：

```python
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(torch.flatten(self.features(x), 1))
```

資料增強**只用在訓練集**；驗證與測試只做固定的 resize／轉張量／與預訓練權重相容的正規化。用 `torchvision.transforms.v2` 的 `RandomHorizontalFlip`、`RandomRotation` 等時，先確認操作不改變標籤意義，例如醫療影像左右方向可能重要。

```python
from torchvision.models import ResNet18_Weights, resnet18

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)  # 首次使用會下載權重
model.fc = nn.Linear(model.fc.in_features, n_classes)
validation_transform = weights.transforms()
```

可先凍結 backbone、只訓練 `model.fc`，確認流程後再微調後段。若必須離線運作，使用 `weights=None` 可檢查架構，但那**不是遷移學習**；需提供已下載的合法權重才算預訓練。`weights.transforms()` 對應權重要求的影像大小與正規化。完整規則見 [torchvision ResNet18 文件](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)。

## 5. 儲存、推論與排錯

```python
torch.save(model.state_dict(), "model.pt")
restored = SmallCNN(n_classes)
restored.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
restored.eval()
```

只載入可信來源的模型檔；保存推論需要的架構版本、類別對照與影像前處理，不要只存權重。建議固定 random seed，並記錄資料版本、切分方式、套件版本、超參數與評估結果。GPU 上部分操作仍可能非完全決定性。

| 症狀 | 先檢查 |
|---|---|
| loss 是 NaN | 資料是否有 NaN/Inf、learning rate、數值尺度、loss/logits 配對 |
| 訓練 accuracy 高、驗證低 | 洩漏、資料分布、模型容量、正則化、增強是否只在訓練集 |
| loss 不下降 | 標籤 dtype/shape、`optimizer.step()`、模型參數是否凍結、learning rate |
| CPU/GPU 裝置錯誤 | model、輸入、標籤是否在同一 `device` |
| BatchNorm／Dropout 推論不穩 | 是否呼叫 `model.eval()`；驗證是否使用 `inference_mode()` |
| 圖像預測變差 | 顏色通道、輸入尺寸、訓練/推論正規化是否一致 |

參考：[PyTorch 基礎教學](https://docs.pytorch.org/tutorials/beginner/basics/)、[訓練迴圈](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)、[模型儲存](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html)、[遷移學習](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)。
