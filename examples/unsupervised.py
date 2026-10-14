"""數值資料的 KMeans、輪廓係數與 PCA 示範。"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    X, _ = make_blobs(n_samples=300, centers=3, n_features=4, cluster_std=0.7, random_state=42)
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=3, n_init="auto", random_state=42))
    labels = model.fit_predict(X)
    scaled = model[:-1].transform(X)
    cluster_ids, sizes = np.unique(labels, return_counts=True)
    print("各群數量:", {int(group): int(size) for group, size in zip(cluster_ids, sizes)})
    print("輪廓係數:", round(silhouette_score(scaled, labels), 3))
    projection = PCA(n_components=2, random_state=42).fit_transform(scaled)
    print("PCA 2D shape:", projection.shape)
    assert len(set(labels)) == 3 and projection.shape == (300, 2)


if __name__ == "__main__":
    main()
