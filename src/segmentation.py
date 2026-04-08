"""
Customer Segmentation using Clustering Algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from yellowbrick.cluster import KElocbow, SilhouetteVisualizer
import warnings
warnings.filterwarnings('ignore')


class CustomerSegmenter:
    """Perform customer segmentation using various clustering algorithms"""
    
    def __init__(self, data):
        """
        Initialize the segmenter
        
        Parameters:
        data: numpy array or pandas DataFrame with features for clustering
        """
        self.data = data
        self.best_k = None
        self.model = None
        self.labels = None
    
    def find_optimal_clusters(self, max_k=10):
        """Use elbow method and silhouette score to find optimal k"""
        k_range = range(2, max_k + 1)
        
        inertias = []
        silhouette_scores = []
        calinski_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.data)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.data, labels))
            calinski_scores.append(calinski_harabasz_score(self.data, labels))
        
        # Plot Elbow Method
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(list(k_range), inertias, 'bo-')
        axes[0].set_xlabel('Number of Clusters (k)')
        axes[0].set_ylabel('Inertia')
        axes[0].set_title('Elbow Method')
        axes[0].grid(True)
        
        # Plot Silhouette Score
        axes[1].plot(list(k_range), silhouette_scores, 'ro-')
        axes[1].set_xlabel('Number of Clusters (k)')
        axes[1].set_ylabel('Silhouette Score')
        axes[1].set_title('Silhouette Score Method')
        axes[1].grid(True)
        
        # Plot Calinski-Harabasz Score
        axes[2].plot(list(k_range), calinski_scores, 'go-')
        axes[2].set_xlabel('Number of Clusters (k)')
        axes[2].set_ylabel('Calinski-Harabasz Score')
        axes[2].set_title('Calinski-Harabasz Score')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('visualizations/elbow_method.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Select best k based on silhouette score
        self.best_k = list(k_range)[np.argmax(silhouette_scores)]
        print(f"Optimal number of clusters: {self.best_k}")
        
        return self.best_k
    
    def kmeans_segmentation(self, n_clusters=None):
        """Perform K-Means clustering"""
        if n_clusters is None:
            if self.best_k is None:
                self.find_optimal_clusters()
            n_clusters = self.best_k
        
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.labels = self.model.fit_predict(self.data)
        
        # Print metrics
        silhouette_avg = silhouette_score(self.data, self.labels)
        calinski_avg = calinski_harabasz_score(self.data, self.labels)
        
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Calinski-Harabasz Score: {calinski_avg:.4f}")
        
        return self.labels
    
    def hierarchical_segmentation(self, n_clusters=None):
        """Perform Hierarchical clustering"""
        if n_clusters is None:
            if self.best_k is None:
                self.find_optimal_clusters()
            n_clusters = self.best_k
        
        self.model = AgglomerativeClustering(n_clusters=n_clusters)
        self.labels = self.model.fit_predict(self.data)
        
        silhouette_avg = silhouette_score(self.data, self.labels)
        print(f"Hierarchical Clustering - Silhouette Score: {silhouette_avg:.4f}")
        
        return self.labels
    
    def profile_segments(self, df_original, labels):
        """Create profile for each segment"""
        df_with_labels = df_original.copy()
        df_with_labels['Segment'] = labels
        
        # Calculate segment statistics
        segment_stats = df_with_labels.groupby('Segment').mean()
        
        # Visualize segment sizes
        plt.figure(figsize=(10, 6))
        segment_sizes = df_with_labels['Segment'].value_counts()
        sns.barplot(x=segment_sizes.index, y=segment_sizes.values, palette='viridis')
        plt.xlabel('Segment')
        plt.ylabel('Number of Customers')
        plt.title('Customer Distribution Across Segments')
        plt.savefig('visualizations/segment_sizes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return segment_stats
    
    def visualize_clusters(self, df_original, labels, x_col, y_col):
        """Visualize clusters in 2D space"""
        df_plot = df_original.copy()
        df_plot['Segment'] = labels
        
        plt.figure(figsize=(10, 8))
        scatter = sns.scatterplot(
            data=df_plot,
            x=x_col,
            y=y_col,
            hue='Segment',
            palette='viridis',
            alpha=0.7,
            s=100
        )
        
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title('Customer Segments Visualization')
        plt.legend(title='Segment')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/cluster_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
