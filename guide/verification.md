# 更新與驗證紀錄

[← 工具書首頁](../README.md)

檢視日期：2026-10-14。程式範例在 Windows、Python 3.11.9、CPU 環境執行；實際套件版本為 NumPy 2.4.6、pandas 3.0.6、scikit-learn 1.9.1、SciPy 1.17.1、matplotlib 3.11.2、seaborn 0.13.2、PyTorch 2.14.1+cpu、torchvision 0.29.1+cpu。新環境依 [requirements](../requirements-modern.txt) 安裝，PyTorch／torchvision 使用 [官方 CPU wheel 索引](https://download.pytorch.org/whl/cpu)。

已執行 `examples/` 的資料整理、分類、回歸、分群與 PyTorch 訓練腳本；7 本新 PyTorch notebook 的程式 cell 逐本執行，並通過 nbformat 結構驗證。4 張 HTML 流程圖通過 Diagram Design 的 `self_check.py`，並以瀏覽器檢查 1200×850 畫面。原 `homework/` 94 本 notebook 通過 JSON 解析，但因多本需要未提供的資料檔、外部網路資源或舊套件，未宣稱原始程式全部可重跑。

這些示例使用合成資料教流程，產生的分數不代表真實業務表現。移植到真實資料時，應重新確認資料粒度、群組／時間切分、類別比例、特徵洩漏、資料權限與業務指標。

## 2026-10-15：TensorFlow 與雙框架增補

另附的 `ML&DL_筆記.md` 已依主題重整為 [TensorFlow](tensorflow.md)、[概念速查](dl-concepts.md)、[雙框架實作對照](framework-compare.md) 與 [筆記勘誤](note-map.md)；舊筆記的外部 HackMD 圖片沒有複製進 repo。依 TensorFlow、Keras、PyTorch 官方文件核對目前 API，尤其是 Windows GPU 支援、`tf.data`、Keras 3 `.keras`／SavedModel 匯出、回呼與遷移學習。

在 Windows、Python 3.11.9、CPU 上使用 TensorFlow 2.21.0／Keras 3.15.1 執行 [TensorFlow](../examples/tf_classification.py) 與 PyTorch 2.14.1+cpu 的 [配對分類](../examples/torch_classification_pair.py)；兩者共用離線資料與切分，驗證 accuracy 均為 1.000。三本 [TensorFlow notebook](../modern/tensorflow/README.md) 的程式 cell 逐格執行通過，含 `.keras` 保存／載入、CNN、GRU 與無預訓練權重的 MobileNetV2 架構檢查；另以獨立小模型實際驗證 `model.export` → `tf.saved_model.load` → `serve`。新流程圖通過 Diagram Design `self_check.py`，以瀏覽器檢視 1200×1100 畫面。全部 94 本歷史 notebook 和 3 本新增 TensorFlow notebook 通過 JSON 解析；工具書相對路徑與章節錨點檢查無斷鏈，`git diff --check` 無空白問題。

上述準確率只反映簡單合成資料與教學流程。尚未驗證原作業外部資料、實際 GPU 訓練、ImageNet 權重下載或真實服務部署；遷移學習 notebook 的 `weights=None` 只檢查架構，並不代表已測預訓練效果。
