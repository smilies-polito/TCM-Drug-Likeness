import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from hdbscan import HDBSCAN
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import itertools
import warnings

# Ignore warnings to keep the output cleaner
warnings.filterwarnings("ignore")

def run_kmeans(n_clusters, X):
    """
    Executes KMeans clustering and computes the Silhouette Score.
    
    Args:
        n_clusters (int): Number of clusters.
        X (np.ndarray): Data to cluster.
    
    Returns:
        dict: Results containing the Silhouette Score.
    """
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Compute the Silhouette Score
        silhouette = silhouette_score(X, cluster_labels)
        
        return {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette
        }
    except Exception as e:
        print(f"Error with KMeans for {n_clusters} clusters: {e}")
        return {
            'n_clusters': n_clusters,
            'silhouette_score': np.nan
        }

def run_hdbscan(min_samples, min_cluster_size, X):
    """
    Executes HDBSCAN clustering and computes the Silhouette Score.
    
    Args:
        min_samples (int): The min_samples parameter for HDBSCAN.
        min_cluster_size (int): The min_cluster_size parameter for HDBSCAN.
        X (np.ndarray): Data to cluster.
    
    Returns:
        dict: Results containing the Silhouette Score.
    """
    print(f"Running HDBSCAN with min_samples={min_samples}, min_cluster_size={min_cluster_size}")
    try:
        hdb = HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric='euclidean')
        cluster_labels = hdb.fit_predict(X)
        clustered = (cluster_labels >= 0)
        
        # Check that at least 75% of the points are clustered
        percent_clustered = (np.sum(clustered) / len(cluster_labels)) * 100
        if percent_clustered < 75:
            print(f"Clustered points below 75%: {percent_clustered:.2f}% for min_samples={min_samples}, min_cluster_size={min_cluster_size}")
            return {
                'min_samples': min_samples,
                'min_cluster_size': min_cluster_size,
                'silhouette_score': -1
            }
        
        # Compute the Silhouette Score
        silhouette = silhouette_score(X[clustered], cluster_labels[clustered])
        
        return {
            'min_samples': min_samples,
            'min_cluster_size': min_cluster_size,
            'silhouette_score': silhouette
        }
    except Exception as e:
        print(f"Error with HDBSCAN for min_samples={min_samples}, min_cluster_size={min_cluster_size}: {e}")
        return {
            'min_samples': min_samples,
            'min_cluster_size': min_cluster_size,
            'silhouette_score': np.nan
        }

def PCA_param_tuning(n_components = None, fp_path = 'molecule_fingerprints2.csv', out_dir = '../results'):
    # 1. Parameters definition
    if n_components is None:
        n_components = 610
        
    n_clusters_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300]
    min_samples_range = [5, 10, 20]
    min_cluster_size_range = [30, 50, 70, 100, 150]
    
    # 2. Separate the features and the molecule ID
    fingerprint = pd.read_csv(fp_path)
    X = fingerprint.drop('id_molecola', axis=1).values
    id_molecule = fingerprint['id_molecola']
    
    # 3. Data preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled).astype(np.float64)
    
    # 5. KMeans clustering
    print("Starting KMeans")
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Launch processes for KMeans
        futures_kmeans = {executor.submit(run_kmeans, n_clusters, X_pca): n_clusters for n_clusters in n_clusters_list}
        kmeans_results = []
        for future in as_completed(futures_kmeans):
            result = future.result()
            kmeans_results.append(result)
    
    # Save KMeans results
    kmeans_df = pd.DataFrame(kmeans_results)
    kmeans_df.to_csv(f'{out_dir}/PCA_KMeans_results.csv', index=False)
    print("Finished KMeans")
    print("Starting HDBSCAN")
    
    # 6. HDBSCAN clustering
    # Prepare parameter combinations
    hdbscan_params = list(itertools.product(min_samples_range, min_cluster_size_range))
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Launch processes for HDBSCAN
        futures_hdbscan = {executor.submit(run_hdbscan, params[0], params[1], X_pca): params for params in hdbscan_params}
        hdbscan_results = []
        for future in as_completed(futures_hdbscan):
            result = future.result()
            hdbscan_results.append(result)
    
    # Save HDBSCAN results
    hdbscan_df = pd.DataFrame(hdbscan_results)
    hdbscan_df.to_csv(f'{out_dir}/PCA_HDBSCAN_results.csv', index=False)
    
    # 7. Visualization of Results
    
    # Plotting metrics for KMeans (commented out)
    # plt.figure(figsize=(10, 6))
    # plt.plot(kmeans_df['n_clusters'], kmeans_df['silhouette_score'], marker='o', linestyle='--', label='Silhouette Score')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Score vs Number of Clusters (KMeans)')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(r'../results/KMeans_Silhouette_Score.png')
    # plt.show()
    
    # 3D plot for HDBSCAN Silhouette Score (commented out)
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(hdbscan_df['min_samples'], hdbscan_df['min_cluster_size'], hdbscan_df['silhouette_score'], c='b', marker='o')
    # ax.set_xlabel('min_samples')
    # ax.set_ylabel('min_cluster_size')
    # ax.set_zlabel('HDBSCAN Silhouette Score')
    # ax.set_title('HDBSCAN Silhouette Score')
    # plt.savefig(r'../results/HDBSCAN_Silhouette_Score.png')
    # plt.show()
    
    # Find the best Silhouette Score for KMeans
    best_kmeans_row = kmeans_df.loc[kmeans_df['silhouette_score'].idxmax()]
    best_kmeans_silhouette = best_kmeans_row['silhouette_score']
    best_kmeans_n_clusters = int(best_kmeans_row['n_clusters'])
    print(f"KMeans -> Best Silhouette Score: {best_kmeans_silhouette} with {best_kmeans_n_clusters} clusters")

    # Find the best Silhouette Score for HDBSCAN
    valid_hdbscan = hdbscan_df[hdbscan_df['silhouette_score'] != -1].dropna(subset=['silhouette_score'])
    if not valid_hdbscan.empty:
        best_hdbscan_row = valid_hdbscan.loc[valid_hdbscan['silhouette_score'].idxmax()]
        best_hdbscan_silhouette = best_hdbscan_row['silhouette_score']
        best_hdbscan_min_sample = best_hdbscan_row['min_samples']
        best_hdbscan_min_cluster_size = best_hdbscan_row['min_cluster_size']
        print(f"HDBSCAN -> Best Silhouette Score: {best_hdbscan_silhouette} with config min_sample={best_hdbscan_min_sample} and min_cluster_size={best_hdbscan_min_cluster_size}")
    else:
        best_hdbscan_silhouette = -1
        best_hdbscan_min_sample = -1
        best_hdbscan_min_cluster_size = -1
        print("No valid silhouette score found for HDBSCAN.")

    # Compare the best results of KMeans and HDBSCAN
    if best_kmeans_silhouette > best_hdbscan_silhouette:
        print(f"The best configuration is KMeans with Silhouette Score: {best_kmeans_silhouette} and {best_kmeans_n_clusters} clusters")
    else:
        print(f"The best configuration is HDBSCAN with Silhouette Score: {best_hdbscan_silhouette}, min_sample={best_hdbscan_min_sample} and min_cluster_size={best_hdbscan_min_cluster_size}")

    # Return the best configuration with its corresponding Silhouette Score
    return best_kmeans_silhouette, best_kmeans_n_clusters, best_hdbscan_silhouette, best_hdbscan_min_sample, best_hdbscan_min_cluster_size
   
# if __name__ == "__main__":
#     PCA_param_tuning(610,'new_molecule_fingerprints2.csv')
