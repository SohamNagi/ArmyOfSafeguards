"""KNN-based aggregator for combining multiple safeguard predictions.

This module implements a K-Nearest Neighbors (KNN) aggregator that uses
reference data to predict safety based on safeguard confidence values.
"""
import numpy as np
from typing import List, Dict, Any
from sklearn.neighbors import NearestNeighbors


class KNNAggregator:
    """KNN aggregator that uses k-nearest neighbors to predict safety based on safeguard confidences."""
    
    def __init__(self, k: int = 7, metric: str = 'euclidean'):
        """
        Initialize KNN aggregator.
        
        Args:
            k: Number of nearest neighbors to consider (default: 7)
            metric: Distance metric for KNN (default: 'euclidean')
        """
        self.k = k
        self.metric = metric
        self.nn = None  # Will be initialized in fit()
        self.is_fitted = False
        self.reference_labels = None

    def fit(self, reference_data: List[Dict[str, Any]]):
        """
        Fit the KNN model on reference data.
        
        Args:
            reference_data: List of dictionaries with format:
                [
                    {
                        'confidences': [0.92, 0.12, 0.88, 0.03],  # factuality, toxicity, sexual, jailbreak
                        'is_safe': False  # True = safe, False = unsafe
                    },
                    ...
                ]
        
        Raises:
            ValueError: If reference_data is empty or invalid
        """
        if not reference_data:
            raise ValueError("reference_data cannot be empty")
        
        # Validate and extract data
        X_list = []
        y_list = []
        
        for i, item in enumerate(reference_data):
            if 'confidences' not in item or 'is_safe' not in item:
                raise ValueError(f"Item {i} missing 'confidences' or 'is_safe' key")
            
            confidences = item['confidences']
            if not isinstance(confidences, list) or len(confidences) != 4:
                raise ValueError(f"Item {i}: confidences must be a list of 4 floats, got {confidences}")
            
            if not isinstance(item['is_safe'], bool):
                raise ValueError(f"Item {i}: is_safe must be a boolean, got {type(item['is_safe'])}")
            
            X_list.append(confidences)
            y_list.append(item['is_safe'])
        
        X = np.array(X_list)  # (N, 4)
        y = np.array(y_list)  # (N,)
        
        # Adjust k if necessary
        actual_k = min(self.k, len(X))
        if actual_k < self.k:
            print(f"[KNN] Warning: k={self.k} but only {len(X)} samples available, using k={actual_k}")
        
        self.nn = NearestNeighbors(n_neighbors=actual_k, metric=self.metric)
        self.nn.fit(X)
        self.reference_labels = y
        self.is_fitted = True
        print(f"[KNN] Fitted with {len(X)} reference samples, k={actual_k}")

    def predict(self, confidences_4d: List[float]) -> Dict[str, Any]:
        """
        Predict safety using KNN on safeguard confidences.
        
        Args:
            confidences_4d: List of 4 confidence values [factuality, toxicity, sexual, jailbreak]
        
        Returns:
            Dictionary with prediction results:
            {
                'is_safe': bool,
                'knn_safe_votes': int,
                'knn_unsafe_votes': int,
                'knn_distances': List[float],
                'method': str
            }
        
        Raises:
            RuntimeError: If model not fitted
            ValueError: If confidences_4d is invalid
        """
        if not self.is_fitted:
            raise RuntimeError("KNN not fitted yet! Call fit() first.")
        
        if not isinstance(confidences_4d, list) or len(confidences_4d) != 4:
            raise ValueError(f"confidences_4d must be a list of 4 floats, got {confidences_4d}")
        
        if not all(isinstance(c, (int, float)) for c in confidences_4d):
            raise ValueError(f"All confidences must be numbers, got {confidences_4d}")

        query = np.array(confidences_4d).reshape(1, -1)
        distances, indices = self.nn.kneighbors(query)

        neighbor_labels = self.reference_labels[indices[0]]
        # Count safe votes (True = 1, False = 0)
        safe_votes = int(np.sum(neighbor_labels))
        actual_k = len(neighbor_labels)
        unsafe_votes = actual_k - safe_votes

        # Majority vote: safe if more than half are safe, otherwise unsafe
        final_safe = safe_votes > unsafe_votes

        return {
            'is_safe': bool(final_safe),
            'knn_safe_votes': safe_votes,
            'knn_unsafe_votes': unsafe_votes,
            'knn_distances': distances[0].tolist(),
            'method': f'knn_{actual_k}'
        }

