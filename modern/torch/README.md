# PyTorch 練習路線

[← 工具書首頁](../../README.md) · [PyTorch 情境工具書](../../guide/pytorch.md) · [TensorFlow 練習](../tensorflow/README.md) · [雙框架對照](../../guide/framework-compare.md) · [原作業索引](../../guide/legacy-index.md)

原 Day 63–100 中的 Keras／TensorFlow 程式保留為歷史筆記，每本舊 notebook 開頭都有新版入口。以下七本是現代教學；預設使用 CPU 可執行的合成資料，避免依賴未附在 repo 的 Kaggle 檔案。

| 先學什麼／原 Day | Notebook | 學完能做什麼 |
|---|---|---|
| 63–70、98：基本元件 | [01 張量、資料、模型](01_tensors_dataset_module.ipynb) | 確認 tensor shape／dtype，建立 `DataLoader` 與 `nn.Module` |
| 71、73–80、85–87：訓練順序 | [02 訓練迴圈](02_training_loop.ipynb) | 正確訓練與驗證，避免在驗證集更新參數 |
| 71–84：優化與過擬合 | [03 損失與正則化](03_loss_optimizer_regularization.ipynb) | 選 loss、SGD／AdamW、Dropout、BatchNorm、L1 |
| 85–89：訓練控制 | [04 排程、早停、自訂 loss](04_training_controls_custom_loss.ipynb) | 保存最佳權重，記錄類別別指標 |
| 92–97：影像模型 | [05 CNN](05_cnn.ipynb) | 理解卷積尺寸與訓練小型 CNN |
| 99–100：影像資料 | [06 增強與遷移學習](06_augmentation_transfer.ipynb) | 使用 `torchvision.transforms.v2`，替換 ResNet 分類頭 |
| 延伸 63：序列 | [07 GRU](07_sequences.ipynb) | 處理不同長度序列，建立輕量序列基準 |

Day 90–91 是顏色直方圖、HOG + SVM 等傳統電腦視覺特徵；請讀 [分類模型選擇](../../guide/ml.md)，不必硬改成神經網路。原 Day 72 的 ReLU 導數、Day 74–75 的梯度概念，可在第 01–03 本的 `loss.backward()` 與參數梯度中對照。

在 repo 根目錄的虛擬環境中執行 notebook；套件安裝方式見 [首頁](../../README.md)。第 06 本需要 `torchvision`，請與 `torch` 使用相容版本。
