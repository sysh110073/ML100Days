# 更新與驗證紀錄

[← 工具書首頁](../README.md)

檢視日期：2026-10-14。程式範例在 Windows、Python 3.11.9、CPU 環境執行；實際套件版本為 NumPy 2.4.6、pandas 3.0.6、scikit-learn 1.9.1、SciPy 1.17.1、matplotlib 3.11.2、seaborn 0.13.2、PyTorch 2.14.1+cpu、torchvision 0.29.1+cpu。新環境依 [requirements](../requirements-modern.txt) 安裝，PyTorch／torchvision 使用 [官方 CPU wheel 索引](https://download.pytorch.org/whl/cpu)。

已執行 `examples/` 的資料整理、分類、回歸、分群與 PyTorch 訓練腳本；7 本新 PyTorch notebook 的程式 cell 逐本執行，並通過 nbformat 結構驗證。4 張 HTML 流程圖通過 Diagram Design 的 `self_check.py`，並以瀏覽器檢查 1200×850 畫面。原 `homework/` 94 本 notebook 通過 JSON 解析，但因多本需要未提供的資料檔、外部網路資源或舊套件，未宣稱原始程式全部可重跑。

這些示例使用合成資料教流程，產生的分數不代表真實業務表現。移植到真實資料時，應重新確認資料粒度、群組／時間切分、類別比例、特徵洩漏、資料權限與業務指標。
