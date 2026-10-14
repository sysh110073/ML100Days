# 沒有標籤：分群、異常偵測與降維

[← 工具書首頁](../README.md) · [無標籤流程圖](diagrams/unsupervised-flow.html) · [可執行範例](../examples/unsupervised.py)

先問：需要的是**可操作的群組**、**尋找稀有異常**，還是**壓縮／視覺化維度**？這三個目的不能用同一個分數回答。沒有真實標籤也能使用內部指標；但群組有沒有業務意義仍需外部驗證。

| 目標／資料形狀 | 候選工具 | 何時合適 | 主要限制 |
|---|---|---|---|
| 預先想要 K 個相近且較圓的群 | `KMeans` | 數值尺度已處理、群大小相近 | 需選 K；對離群點敏感 |
| 群的密度不同、想辨識噪聲 | `DBSCAN`、`HDBSCAN`（若環境支援） | 非球形群／噪聲 | 距離尺度與參數敏感；高維資料困難 |
| 要看巢狀關係 | `AgglomerativeClustering`、`scipy.cluster.hierarchy` | 樣本數不大、需要階層圖 | 大量資料可能昂貴 |
| 想找少數異常紀錄 | `IsolationForest`、`LocalOutlierFactor` | 異常分數／候選清單 | 分數不等於錯誤或詐欺，需驗證 |
| 線性壓縮、保留主要變異 | `PCA`、`TruncatedSVD`（稀疏矩陣） | 降噪、壓縮、前處理 | 最大變異不一定是目標訊號 |
| 高維資料的 2D／3D 探索 | `TSNE`、UMAP（額外套件） | 畫局部結構、發現候選型態 | 圖上距離／群大小不可直接當原空間結論；不拿 t-SNE 當正式預測器 |

## 分群的基本流程

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=3, n_init="auto", random_state=42))
labels = pipeline.fit_predict(X_numeric)
score = silhouette_score(pipeline[:-1].transform(X_numeric), labels)
print(score)
```

輪廓係數可輔助比較 K，但只對有至少 2 群、且群數少於樣本數的結果有定義。它偏好某些幾何形狀，不能取代群的穩定性與可解釋性檢查。若要預測新樣本所屬群，`KMeans` 有 `predict()`；`AgglomerativeClustering` 一般沒有直接的新樣本預測介面。

## PCA 與 t-SNE 分別做什麼

```python
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

X_scaled = StandardScaler().fit_transform(X_numeric)
X_pca = PCA(n_components=0.95, random_state=42).fit_transform(X_scaled)
X_map = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X_scaled)
```

上例是探索式用法。若 PCA 是**監督模型的一部分**，應把 `StandardScaler`、`PCA` 和模型一起放進 Pipeline，再做交叉驗證，避免測試資料影響主成分。t-SNE 的 `perplexity` 必須小於樣本數；先在合理樣本量上試多個種子／參數，觀察視覺結構是否穩定。把 2D 散點的可見「空白」直接解讀為真實類別邊界是不可靠的。

## 怎麼評估沒有標籤的結果

1. **技術檢查**：群大小、噪聲比例、輪廓係數、不同種子與抽樣的穩定性。
2. **領域檢查**：群的特徵分布是否有可理解差異；是否對後續決策有用。
3. **若後來取得可信標籤**：可用 ARI、NMI 等比較，但那是外部標籤評估，不應再說「完全無標籤」。
4. **異常偵測**：用人工覆核或後續事件確認 precision／recall；先定義可處理的警報量。

參考：[scikit-learn 分群](https://scikit-learn.org/stable/modules/clustering.html)、[PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)、[t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne)、[異常偵測](https://scikit-learn.org/stable/modules/outlier_detection.html)。
