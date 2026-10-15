# 深度學習概念速查：從原筆記到可用判斷

[← 首頁](../README.md) · [TensorFlow 實作](tensorflow.md) · [PyTorch 實作](pytorch.md) · [雙框架對照](framework-compare.md) · [來源主題](note-map.md)

本章把 `ML&DL_筆記.md` 的激勵函式、優化、CNN 演進與序列模型放在同一條判斷路徑。公式說明的是**何時採用、哪些形狀必須對上**；實際訓練先用 [TensorFlow 分類腳本](../examples/tf_classification.py) 或 [PyTorch 配對腳本](../examples/torch_classification_pair.py)。

## 1. 輸出目標決定 loss

| 輸出問題 | 常用輸出／loss | 什麼時候檢查別的指標 |
|---|---|---|
| 連續數值 | 線性輸出 + MAE／MSE | 極端誤差成本高時比較 RMSE；不宜只看 loss |
| 二類互斥 | 1 logit + binary cross entropy | 類別不平衡時看 precision、recall、PR-AUC、決策門檻 |
| 多類互斥 | `C` logits + sparse／categorical cross entropy | 依標籤是索引或 one-hot 選對應 loss |
| 多標籤可同時成立 | 每類獨立 logit + binary cross entropy | 每類的閾值與代價可不同 |

交叉熵衡量真實標籤分布下模型預測的負對數機率；用自然對數或以 2 為底，只是單位與固定倍率不同，並非因電腦用二進位就必須選底數 2。KL 散度 $D_{KL}(P\|Q)$ 比較兩個分布，**不對稱**；當真實分布 $P$ 固定時，交叉熵可分解為 $H(P)+D_{KL}(P\|Q)$，因此最小化交叉熵等價於最小化該 KL 項。對單筆分類的 one-hot 目標，可以直接使用相應交叉熵，不必手刻 KL。[Keras loss API](https://keras.io/api/losses/probabilistic_losses/)

## 2. 隱藏層激勵與梯度

| 函式 | 典型用途 | 注意 |
|---|---|---|
| `ReLU` | MLP／CNN 隱藏層預設起點 | 負半軸梯度為 0；若大量神經元不再更新，檢查學習率／初始化，必要時試 LeakyReLU |
| `LeakyReLU` | 希望負值仍有小梯度 | 斜率是超參數，需驗證 |
| `sigmoid` | 二元／多標籤**推論機率**；門控 | 飽和區梯度小，深層隱藏層通常不以它為預設 |
| `tanh` | RNN 等需有正負輸出的內部狀態 | 也可能飽和 |
| `softmax` | 多類互斥 logits 轉成總和為 1 的機率 | 若 loss 設 `from_logits=True`，訓練輸出先不加它 |
| Maxout | 比較多種線性回應，取逐元素最大值 | 參數／運算成本較高，通常先用 ReLU 基準 |

非線性激勵通常**放在隱藏層**，使多層組合能表示非線性關係。ReLU 在 0 處的數學導數不唯一，框架使用約定的次梯度；不能因此說激勵函式都不可微、只能放最後。梯度消失與爆炸也和深度、權重、正規化、殘差及序列長度有關，不應只歸因於 sigmoid。

## 3. 優化器與訓練控制

1. **先確認 loss 能下降**：核對資料尺度、標籤／logits、是否更新參數，並嘗試讓模型在極小訓練集上過擬合。
2. **基準**：小批次 SGD（可加 momentum）或 Adam／AdamW；記錄 learning rate、batch size 與 seed。
3. **震盪或不收斂**：先調 learning rate，再考慮梯度裁剪、初始化與資料正規化。RMSprop 的自適應步長不是固定的「每個 epoch 都下降」。
4. **驗證不進步**：在固定切分上用早停、降低學習率，並保存**最佳驗證**模型。不要用測試集挑 epoch。
5. **過擬合**：先排除資料洩漏，再比較增強、Dropout、權重正則化、較小模型與更多資料。

`Dropout` 與 BatchNorm 的訓練／推論行為不同，所以 TensorFlow 要傳 `training=False`，PyTorch 要 `model.eval()`。`AdamW` 的解耦 weight decay 與加在 loss 上的 L2 懲罰不可在所有設定下視作相同。若原筆記以一次隨機切分對比多個 optimizer，該結論仍需重複實驗與相同資料處理才能解讀。

## 4. CNN：何時卷積、怎麼算 shape

卷積用**局部連接與權重共享**控制參數量，適用於影像，也可用於有局部模式的序列。它是一種數學歸納偏好，不能簡化為「模仿人眼」。單層輸出高／寬可用下式檢查：

$$H_{out}=\left\lfloor\frac{H_{in}+2P-D(K-1)-1}{S}+1\right\rfloor$$

$K$ 是 kernel 大小、$S$ stride、$P$ 兩側 padding、$D$ dilation。若採 `same`，常見框架對輸出尺寸採 $\lceil H_{in}/S\rceil$；**stride 2 仍會縮小**。一層普通卷積的可訓練參數數量是 $K_hK_wC_{in}C_{out}+C_{out}$（有 bias 時）。TensorFlow 核心權重 shape 通常 `[Kh,Kw,Cin,Cout]`；PyTorch 是 `[Cout,Cin,Kh,Kw]`。池化一般不增加可訓練權重；Global Average Pooling 把每張 feature map 壓成一個值，降低分類頭的參數量。

| 架構概念 | 要學的取捨 | 今日實作入口 |
|---|---|---|
| LeNet／AlexNet | 卷積、池化、全連接的基本組合 | 先做小 CNN，印每層 shape |
| VGG | 多個小 kernel 堆疊增加感受野／深度 | 作架構理解；部署時考慮大小與成本 |
| GoogLeNet／Inception | 平行分支、1×1 卷積調整通道，GAP／輔助輸出 | 用 Functional API 或自訂 Module 研究分支；一般任務先試預訓練模型 |
| ResNet | 殘差路徑協助深層訓練；shape 不同時可能需投影 | [Keras Applications](https://keras.io/api/applications/)／[torchvision models](https://docs.pytorch.org/vision/stable/models.html) |

架構名字不是模型選擇的答案。少量影像先試合適的預訓練權重；關注任務類別、前處理一致性、運算成本與驗證表現。原筆記一些卷積示意圖的通道／節點敘述把**每筆特徵數**與**batch 張數**混在一起，應每層都明確寫 `[batch, spatial, channels]` 或 `[batch, channels, spatial]`。

## 5. 序列：RNN → GRU／LSTM → 注意力

序列資料最先要決定「預測時可見到哪些時間點」與「目標對應哪個時間點」。RNN 用前一步狀態更新下一步，GRU／LSTM 透過門控調節資訊保留；長序列與大資料可比較注意力／Transformer 或可用的預訓練模型。

| 情境 | 起點 | 資料形狀／風險 |
|---|---|---|
| 固定長度、短時間依賴 | 手工 lag 特徵 + 線性／樹模型 | 先與上一值、移動平均、季節性基準比較 |
| 變長序列分類 | Embedding + GRU／LSTM | `[N,T,F]`；padding 需 mask，不能把補值當真實時間步 |
| 文字理解／生成 | 合適的預訓練模型，或小型注意力模型作教學 | 分詞器與模型版本要匹配；生成需因果 mask |
| 對齊輸入／輸出序列 | Encoder–Decoder + cross-attention | 輸入 padding mask、輸出因果 mask 各有用途 |

矩陣記憶法 $x_tW_{xh}+h_{t-1}W_{hh}$ 有助理解維度：若用**列向量右乘**，$W_{xh}$ 可寫 `[input, hidden]`、$W_{hh}$ 是 `[hidden, hidden]`。實際框架儲存 shape 可轉置或拼接多個門，不能直接照圖複製權重。Keras `GRU(return_sequences=True)` 回傳每一時間步；PyTorch `GRU(batch_first=True)` 會回傳 `(outputs, last_hidden)`。[Keras GRU](https://keras.io/api/layers/recurrent_layers/gru/) · [PyTorch GRU](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html)

**Self-attention** 讓位置參照序列中的其他可見位置。Encoder 的雙向注意力適合理解輸入；自回歸 decoder 用因果遮罩避免看未來 token；Encoder–Decoder 另有 cross-attention 連接輸入與輸出。實際選型依任務與已有模型，不以「最新架構」替代資料切分、成本與品質評估。[原始 Transformer 論文](https://arxiv.org/abs/1706.03762) · [Keras MultiHeadAttention](https://keras.io/api/layers/attention_layers/multi_head_attention/) · [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
