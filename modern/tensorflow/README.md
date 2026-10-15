# TensorFlow 練習路線

[← 工具書首頁](../../README.md) · [TensorFlow 情境教學](../../guide/tensorflow.md) · [TensorFlow ↔ PyTorch](../../guide/framework-compare.md) · [來源筆記對照](../../guide/note-map.md)

三本 notebook 依另附的 `ML&DL_筆記.md` 整理，預設只用 CPU 與離線合成資料，不需 Kaggle 檔或下載權重。先在 repo 根目錄建立獨立環境並安裝 [`requirements-tensorflow.txt`](../../requirements-tensorflow.txt)，再用支援 Jupyter 的編輯器開啟。原筆記的歷史截圖與舊版 CUDA 操作不是新範例的執行前提。

| 順序 | Notebook | 學會什麼 | 可對照的 PyTorch 內容 |
|---|---|---|---|
| 1 | [資料、模型與 fit](01_data_model_fit.ipynb) | `tf.data`、`Sequential`、logits／loss、訓練與驗證 | [張量與資料](../torch/01_tensors_dataset_module.ipynb)、[訓練迴圈](../torch/02_training_loop.ipynb) |
| 2 | [GradientTape、回呼與保存](02_gradient_callbacks_save.ipynb) | 自訂更新步驟、正則化、早停、最佳 `.keras` 模型 | [損失與正則化](../torch/03_loss_optimizer_regularization.ipynb)、[訓練控制](../torch/04_training_controls_custom_loss.ipynb) |
| 3 | [CNN、GRU、遷移架構](03_cnn_gru_transfer.ipynb) | 影像／序列 shape、小型 CNN、GRU、預訓練 backbone 入口 | [CNN](../torch/05_cnn.ipynb)、[增強與遷移](../torch/06_augmentation_transfer.ipynb)、[GRU](../torch/07_sequences.ipynb) |

想快速比較「同一份資料」的兩套實作，直接執行 [TensorFlow](../../examples/tf_classification.py) 與 [PyTorch](../../examples/torch_classification_pair.py) 分類腳本，並讀 [八步速查](../../guide/framework-compare.md#1-速查八個步驟)。真實資料要按時間／群組與推論時點選切分方式，不能照搬合成資料的隨機切分。
