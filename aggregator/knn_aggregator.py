# knn_aggregator.py
import numpy as np
from typing import List, Dict, Any
from sklearn.neighbors import NearestNeighbors

class KNNAggregator:
    def __init__(self, k: int = 7, metric: str = 'euclidean'):
        self.k = k
        self.metric = metric
        self.nn = NearestNeighbors(n_neighbors=k, metric=metric)
        self.is_fitted = False

    def fit(self, reference_data: List[Dict[str, Any]]):
        """
        reference_data: 你验证集跑出来的结果，格式示例：
        [
            {
                'confidences': [0.92, 0.12, 0.88, 0.03],   # factuality, toxicity, sexual, jailbreak 顺序
                'is_safe': False                           # 真实标签（你们手动标注或已有gold label）
            },
            ...
        ]
        """
        X = np.array([item['confidences'] for item in reference_data])  # (N, 4)
        y = np.array([item['is_safe'] for item in reference_data])       # (N,)

        self.nn.fit(X)
        self.reference_labels = y
        self.is_fitted = True
        print(f"[KNN] Fitted with {len(X)} reference samples, k={self.k}")

    def predict(self, confidences_4d: List[float]) -> Dict[str, Any]:
        """
        输入四个模型的 confidence，返回最终聚合结果
        """
        if not self.is_fitted:
            raise RuntimeError("KNN not fitted yet!")

        query = np.array(confidences_4d).reshape(1, -1)
        distances, indices = self.nn.kneighbors(query)

        neighbor_labels = self.reference_labels[indices[0]]
        safe_votes = sum(neighbor_labels)
        unsafe_votes = self.k - safe_votes

        final_safe = safe_votes > unsafe_votes  # 平局算 unsafe 也可以改

        return {
            'is_safe': final_safe,
            'knn_safe_votes': int(safe_votes),
            'knn_unsafe_votes': int(unsafe_votes),
            'knn_distances': distances[0].tolist(),
            'method': f'knn_{self.k}'
        }

