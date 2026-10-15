# `ML&DL_筆記.md` → 工具書主題與勘誤

[← 首頁](../README.md) · [TensorFlow 教學](tensorflow.md) · [雙框架速查](framework-compare.md) · [舊作業遷移](migration.md)

本章記錄使用者另附的 `ML&DL_筆記.md` 如何融入工具書。原筆記保留作學習歷程，包含大量外部 HackMD 截圖；這裡以**可搜尋文字、可複製程式與當前 API** 重整其概念，沒有把截圖誤當成可執行程式。更新檢視：2026-10-15。

| 原筆記主題 | 新入口 | 使用時先問 |
|---|---|---|
| Python、TensorFlow、CUDA、Jupyter 環境 | [TensorFlow 安裝](tensorflow.md#安裝與裝置) | CPU 或 GPU？作業系統與官方支援路徑？ |
| 激勵函式、全連接層、Sequential／Subclassing | [激勵函式判斷](dl-concepts.md)、[TensorFlow API 層級](tensorflow.md#1-何時使用哪一層-api) | 是否只要標準單一路徑模型？ |
| MSE、二元／多類交叉熵、one-hot、熵／KL 散度 | [概念速查](dl-concepts.md)、[loss 對照](tensorflow.md#loss標籤輸出要配對) | 標籤是索引、one-hot、多標籤或連續值？ |
| 梯度、SGD、RMSprop、Adam | [GradientTape](tensorflow.md#3-想掌控每個-batchgradienttape)、[逐步雙框架](framework-compare.md#2-同一個-batch-的訓練步驟) | 是否真的需要自訂更新規則？ |
| `tf.data`、Keras `fit`、EarlyStopping、ModelCheckpoint | [資料流程](tensorflow.md#2-最小分類流程資料--模型--loss--fit)、[回呼與儲存](tensorflow.md#4-過擬合回呼與儲存) | 驗證集有無洩漏？保存最佳還是最後權重？ |
| Dropout、L2、BatchNorm | [過擬合處理](tensorflow.md#4-過擬合回呼與儲存)、[雙框架對照](framework-compare.md#5-回呼遷移學習與部署) | 先排除切分／前處理錯誤；train 與 inference 模式？ |
| CNN、卷積、池化、GAP、LeNet／AlexNet／VGG／GoogLeNet／ResNet | [CNN 概念與架構](dl-concepts.md)、[影像教學](tensorflow.md#5-影像cnn增強與預訓練)、[CNN 雙框架骨架](framework-compare.md#4-架構骨架影像與序列) | 先用小 CNN 還是預訓練 backbone？ |
| Embedding、RNN、LSTM、GRU、注意力、Transformer | [序列與注意力概念](dl-concepts.md)、[序列教學](tensorflow.md#6-序列embeddinggrulstm注意力)、[GRU 雙框架骨架](framework-compare.md#4-架構骨架影像與序列) | 時間切分、padding mask、可用預訓練模型？ |
| 預訓練權重、遷移學習 | [TensorFlow 遷移](tensorflow.md#5-影像cnn增強與預訓練)、[PyTorch 遷移](pytorch.md#4-cnn資料增強與遷移學習) | 凍結 backbone 後是否重新編譯／配置 optimizer？ |
| Model Serving | [部署對照](framework-compare.md#5-回呼遷移學習與部署) | 保存訓練狀態還是提供推論 API？ |

## 需要修正的舊觀念

| 原筆記說法 | 現代解釋 |
|---|---|
| 「激勵函式不可微，不能放神經網路裡面」 | ReLU、sigmoid、tanh 等正是常見**隱藏層**激勵函式。ReLU 在 0 處不可微，框架可採用約定的次梯度，不影響一般反向傳播。輸出層激勵函式要依 loss 與任務選擇。 |
| 「權重不要從 0 開始，因為訓練很久」 | 同層所有權重同值會造成對稱性，使神經元難以學不同特徵；合適初始化的重點是打破對稱並維持訊號尺度。 |
| 「RMSprop 讓學習率越跑越小」 | RMSprop 依梯度平方的移動平均調整**每個參數的有效步長**，不保證整體學習率逐步單調下降。 |
| 「`fit` 一定把資料全部放 RAM，`fit_generator` 才能逐批」 | `fit` 可接受 `tf.data.Dataset` 等逐批輸入；`from_tensor_slices` 包住現成陣列時，陣列本身仍在記憶體。 |
| 「`compile=False` 使用原模型 optimizer／loss」 | 載入時不編譯；要用 `fit` 需再 `compile`。想保留可續訓的完整狀態，使用受支援的 `.keras` 格式並正常載入。 |
| 「`same` padding 的 stride 一定要 1」 | `same` 可搭配 stride > 1，此時輸出空間尺寸會按步幅縮小。 |
| 「3 通道就是三個一模一樣的 filter」 | 每個輸出通道的卷積核會在輸入通道上學各自的權重，然後相加。 |
| 「Flatten 三張 2×2 feature map 等於 12 個節點」 | 若**每筆影像**有 3 個 2×2 feature map，Flatten 後每筆是 12 個特徵；batch 裡幾張影像仍是另一個維度。 |
| 「GAP 一定會犧牲準確率」 | GAP 可減少參數與過擬合風險；效能需在同一驗證方案下實測。 |
| 「scikit-learn 沒有 Model Serving」 | scikit-learn 模型也可包進服務 API；框架是否附帶特定 server，不等於模型能否部署。 |
| 「Windows 加 CUDA 路徑、複製 cuDNN 即可讓現代 TensorFlow 用 GPU」 | 原生 Windows GPU 支援止於 TensorFlow 2.10；新版本採官方 Linux／WSL2 安裝路徑。 |
| 「股票波動大就先全資料歸一化」 | 時間序列須先按時間分訓練／驗證／測試，再只用訓練期估計縮放參數；全資料擬合會洩漏未來。 |

CNN 與序列架構的歷史介紹仍有學習價值，但特定論文架構、圖中的參數量及某些長短處需要依原論文與實際實作核對；本書用可執行小模型說明核心機制，再連到官方預訓練 API。來源筆記的圖片是外部連結，是否持續可用不受本 repo 控制。

主要核對依據：[TensorFlow 安裝](https://www.tensorflow.org/install/pip) · [tf.data](https://www.tensorflow.org/guide/data) · [Keras 保存](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/) · [Keras 匯出](https://keras.io/api/models/model_saving_apis/export/) · [Keras GRU](https://keras.io/api/layers/recurrent_layers/gru/) · [Keras 遷移學習](https://keras.io/guides/transfer_learning/)。
