import json
import os
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import umap
import hdbscan
from hdbscan import validity
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

def _save_clustering_summary(out_dir, red_method, cluster_method, params, **metrics):
    """Salva su disco i parametri e gli indici di qualita' del clustering.

    Erano finora solo stampati a schermo: persistendoli restano disponibili per il confronto
    fra run (anche su macchine diverse) e per la stesura dei risultati.
    """
    summary = {'reduction_method': red_method, 'cluster_method': cluster_method,
               'params': {k: (float(v) if isinstance(v, np.floating) else v)
                          for k, v in params.items()}}
    for k, v in metrics.items():
        summary[k] = float(v) if isinstance(v, (np.floating, np.integer)) else v

    path = os.path.join(out_dir, f'final_{red_method}_{cluster_method}_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Clustering summary saved in {path}")


def hdbscan_clustering(clusterable_embedding, params):
    # Application of HDBSCAN

    min_samples = params['min_samples']
    min_cluster_size = params['min_cluster_size']

    clusterer = hdbscan.HDBSCAN(
        min_samples=params['min_samples'],
        min_cluster_size=params['min_cluster_size'],
        metric='euclidean',  # You can change the metric if needed
        prediction_data=False,
        core_dist_n_jobs=1,  # di default e' 4 (joblib): multi-thread introduce non-determinismo
                             # nel calcolo delle core distance, vicino ai bordi dei cluster puo'
                             # cambiare il numero di cluster risultanti da una run all'altra
    )
    labels = clusterer.fit_predict(clusterable_embedding)

    labels_df = pd.DataFrame(labels, columns=['cluster'])
    
    # Calculate the number of clusters (excluding noise)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # Calculate the percentage of clustered points
    clustered = (labels >= 0)
    tot = clustered.sum() / len(labels) * 100

    # Calculate the Silhouette Score
    if num_clusters > 1:
        silhouette = silhouette_score(clusterable_embedding[clustered], labels[clustered])
    else:
        silhouette = np.nan

    # Calculate the DBCV index
    if num_clusters > 1:
        try:
            dbcv = validity.validity_index(clusterable_embedding[clustered], labels[clustered])
        except AttributeError as ae:
            print(f"Error during DBCV calculation for parameters {params}: {ae}")
            # Print all attributes of the clusterer object for diagnostics
            print(f"HDBSCAN attributes: {dir(clusterer)}")
            dbcv = np.nan
    else:
        print(f"Number of clusters <= 1 for parameters {params}, unable to calculate DBCV.")
        dbcv = np.nan
    
    return labels_df, num_clusters, tot, silhouette, dbcv, clusterer

def run_kmeans(clusterable_embeddings, params):
    
    n_clusters = params['n_clusters']
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    labels = clusterer.fit_predict(clusterable_embeddings)
    
    # Save the labels to a CSV
    labels_df = pd.DataFrame(labels, columns=['cluster'])
    
    # Calculate the Silhouette Score
    silhouette = silhouette_score(clusterable_embeddings, labels)
    
    return labels_df, silhouette, clusterer

def apply_clustering(red_method, cluster_method, fp_path, label_output_dir, params):
    fingerprint = pd.read_csv(fp_path)
    X = fingerprint.drop('id_molecola', axis=1).values
    id_molecules = fingerprint['id_molecola']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clusterer = None

    if red_method == 'PCA':
        print("PCA as reduction method")
        pca = PCA(n_components=params['n_pc'], random_state=42)
        X_red = pca.fit_transform(X_scaled).astype(np.float64)
        # Create a dataframe with the principal components columns and 'id_molecola' from id_molecules and save it as a csv
        clusterable_embedding_df = pd.DataFrame(X_red, columns=[f'pc{i}' for i in range(1, params['n_pc'] + 1)])
        clusterable_embedding_df['id_molecola'] = id_molecules
        

    elif red_method == 'UMAP':
        print("UMAP as reduction method")
        umap_red = umap.UMAP(n_neighbors=params["n_neighbors"], n_components=params['n_components'], min_dist=params["min_dist"], metric='cosine', random_state=42, n_jobs=1)
        X_red = umap_red.fit_transform(X_scaled).astype(np.float64)
        # Create a dataframe with the UMAP columns and 'id_molecola' from id_molecules and save it as a csv
        clusterable_embedding_df = pd.DataFrame(X_red, columns=[f'umap{i}' for i in range(1, params['n_components'] + 1)])
        clusterable_embedding_df['id_molecola'] = id_molecules

    else:
        raise ValueError('Invalid reduction method')

    if cluster_method == 'KMeans':
        print("KMeans as clustering method")
        labels_df, silhouette, clusterer = run_kmeans(X_red, params)
        labels_df.to_csv(os.path.join(label_output_dir, f'final_{red_method}_KMeans_labels_n_clusters_{params["n_clusters"]}.csv'), index=False)
        print(f"Silhouette score: {silhouette}")
        print(f"Number of clusters: {params['n_clusters']}")

        _save_clustering_summary(label_output_dir, red_method, cluster_method, params,
                                 num_clusters=params['n_clusters'], silhouette=silhouette,
                                 n_points=len(labels_df))
        

    elif cluster_method == 'HDBSCAN':
        print("HDBSCAN as clustering method")
        labels_df, num_clusters, tot, silhouette, dbcv, clusterer = hdbscan_clustering(X_red, params)
        labels_df.to_csv(os.path.join(label_output_dir, f'final_{red_method}_HDBSCAN_labels_min_samples_{params["min_samples"]}_min_cluster_size_{params["min_cluster_size"]}.csv'), index=False)

        print(f"Number of clusters: {num_clusters}")
        print(f"Percentage of clustered points: {tot:.2f}%")
        print(f"Silhouette score: {silhouette}")
        print(f"DBCV: {dbcv}")

        _save_clustering_summary(label_output_dir, red_method, cluster_method, params,
                                 num_clusters=num_clusters, clustered_pct=tot,
                                 silhouette=silhouette, dbcv=dbcv, n_points=len(labels_df))

    else:
        raise ValueError('Invalid cluster method')

    # L'embedding ridotto viene persistito insieme alle etichette: e' l'input effettivo del
    # clustering, necessario per riprodurre le analisi a valle (visualizzazione dei cluster,
    # ricalcolo degli indici di qualita') senza dover rieseguire UMAP.
    embedding_path = os.path.join(label_output_dir,
                                  f'final_{red_method}_embedding.csv')
    clusterable_embedding_df.to_csv(embedding_path, index=False)
    print(f"Reduced embedding saved in {embedding_path}")

    return labels_df, clusterer, clusterable_embedding_df

def centroids_extraction(predictions_file, clusterer, clusterable_embedding_df, out_dir):
    """
    Extracts centroids or medoids from clustering results and saves relevant data.

    Parameters:
    - predictions_file: Path to the CSV file with model predictions and SMILES strings.
    - clusterer: Clustering object (e.g., KMeans or HDBSCAN).
    - clusterable_embedding_df: DataFrame with molecule embeddings (must include 'id_molecola').
    - out_dir: Directory to save output files.

    Returns:
    - centroids_df: DataFrame mapping cluster IDs to 'id_molecola' of centroids/medoids.
    """
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    
    # Drop 'id_molecola' column to get the embedding array
    clusterable_embedding = clusterable_embedding_df.drop('id_molecola', axis=1).values

    # Perform clustering
    labels = clusterer.fit_predict(clusterable_embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise (-1)

    # Load predictions and add molecule IDs
    df_predictions = pd.read_csv(predictions_file)
    df_predictions['id_molecola'] = range(len(df_predictions))

    centroids = []

    for i in range(n_clusters):
        print(f"Processing Cluster {i}")
        
        if isinstance(clusterer, KMeans): 
            print("KMeans clustering detected")
            # Extract the centroid
            centroid = clusterer.cluster_centers_[i] 
            
            # CENTROIDS ARE NOT GUARANTEED TO BE IN THE DATA, SO WE NEED TO FIND THE CLOSEST POINT

            # Find the row closest to the centroid in clusterable_embedding_df
            cluster_points = clusterable_embedding[labels == i]  # Points in the cluster
            if cluster_points.shape[0] == 0:  # Skip empty clusters
                print(f"Cluster {i} is empty, skipping...")
                continue

            
            # pairwise_distances_argmin_min restituisce un indice relativo a cluster_points
            # (il sottoinsieme filtrato), non al dataframe completo: va rimappato sugli indici
            # originali, altrimenti si estrae la molecola sbagliata.
            cluster_indices = np.flatnonzero(labels == i)
            closest_index = pairwise_distances_argmin_min([centroid], cluster_points)[0][0]
            id_molecola = clusterable_embedding_df.iloc[cluster_indices[closest_index], -1]
        
        else:  # For HDBSCAN
            print("HDBSCAN clustering detected")
            # Extract the weighted medoid
            centroid = clusterer.weighted_cluster_medoid(i)  
            
            # Find the exact match for the medoid
            matching_rows = clusterable_embedding_df.loc[
                clusterable_embedding_df.iloc[:, :-1].apply(lambda row: np.array_equal(row, centroid), axis=1),
                'id_molecola'
            ]
            if matching_rows.empty:  
                raise ValueError(f"No match found for the medoid in cluster {i}: {centroid}")
            id_molecola = matching_rows.values[0]

        print(f"Centroid/Medoid for Cluster {i}: Molecule ID = {id_molecola}")
        centroids.append(id_molecola)

    # Save centroids to a CSV
    centroids_df = pd.DataFrame({'id_molecola': centroids, 'id_cluster': range(len(centroids))})
    centroids_df.to_csv(f"{out_dir}/centroids_id_molecola.csv", index=False)

    # Save centroid images
    os.makedirs(f"{out_dir}/centroids", exist_ok=True)
    centroid_embeddings = []

    for id_molecola in centroids:
        # Extract embedding for the centroid
        centroid_embedding = clusterable_embedding_df.loc[
            clusterable_embedding_df['id_molecola'] == id_molecola, 
            clusterable_embedding_df.columns[:-1]
        ].values[0]
        centroid_embeddings.append(centroid_embedding)

        # Generate and save molecule image
        smiles = df_predictions.loc[df_predictions['id_molecola'] == id_molecola, 'smiles'].values[0]
        mol = Chem.MolFromSmiles(smiles)
        img = Draw.MolToImage(mol, size=(500, 500))
        img.save(f"{out_dir}/centroids/centroid_{id_molecola}.png")

    # Save centroid embeddings to CSV
    embedding_columns = clusterable_embedding_df.columns[:-1]
    centroid_embeddings_df = pd.DataFrame(centroid_embeddings, columns=embedding_columns)
    centroid_embeddings_df['id_molecola'] = centroids
    centroid_embeddings_df.to_csv(f"{out_dir}/centroid_embeddings.csv", index=False)

    return centroids_df


def test_clustering(fp_path, label_output_dir):
    # Combination 1: PCA + KMeans
    print("Combination 1: PCA + KMeans")
    print("################################# \n\n\n")
    params_pca_kmeans = {
        'n_pc': 2,
        'n_clusters': 3
    }
    apply_clustering('PCA', 'KMeans', fp_path, label_output_dir, params_pca_kmeans)

    # Combination 2: PCA + HDBSCAN
    print("Combination 2: PCA + HDBSCAN")
    print("################################# \n\n\n")
    params_pca_hdbscan = {
        'n_pc': 2,
        'min_samples': 5,
        'min_cluster_size': 10
    }
    apply_clustering('PCA', 'HDBSCAN', fp_path, label_output_dir, params_pca_hdbscan)

    # Combination 3: UMAP + KMeans
    print("Combination 3: UMAP + KMeans")
    print("################################# \n\n\n")
    params_umap_kmeans = {
        "min_dist": 0.1,
        'n_neighbors': 15,
        'n_components': 2,
        'n_clusters': 3
    }
    apply_clustering('UMAP', 'KMeans', fp_path, label_output_dir, params_umap_kmeans)

    # Combination 4: UMAP + HDBSCAN
    print("Combination 4: UMAP + HDBSCAN")
    print("################################# \n\n\n")
    params_umap_hdbscan = {
        'n_neighbors': 15,
        "min_dist": 0.1,
        'n_components': 2,
        'min_samples': 5,
        'min_cluster_size': 10
    }
    apply_clustering('UMAP', 'HDBSCAN', fp_path, label_output_dir, params_umap_hdbscan)

    print("End of clustering methods test")





