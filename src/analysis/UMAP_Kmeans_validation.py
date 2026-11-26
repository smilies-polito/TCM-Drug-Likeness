import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import umap
from hdbscan import validity  # Import for DBCV
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import argparse
import logging
import itertools

def setup_logging():
    """
    Configures logging for the program.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def process_param(params, embeddings_dir, fingerprints_file, id_molecoles, label_output_dir="label_output"):
    """
    Processes a single set of parameters:
    - Checks if the pre-calculated embedding exists.
    - If it exists, loads it.
    - If it does not exist, loads the fingerprints, scales the data, computes UMAP, saves the embedding.
    - Executes KMeans.
    - Calculates evaluation metrics (Silhouette and DBCV).
    - Saves the resulting labels.
    - Returns the metrics results.

    Args:
        params (tuple): Tuple containing (n_neighbors, min_dist, n_components, n_clusters).
        embeddings_dir (str): Directory containing the pre-calculated embeddings.
        fingerprints_file (str): Path to the CSV file containing the original fingerprints of the molecules.
        id_molecoles (pandas.Series): IDs of the molecules.
        label_output_dir (str): Directory to save the labels.

    Returns:
        dict: Metrics results and file paths.
    """
    setup_logging()
    try:
        n_neighbors, min_dist, n_components, n_clusters = params

        # Construct the embedding file name
        embedding_filename = os.path.join(
            embeddings_dir, 
            f'clusterable_embedding_n{n_neighbors}_d{min_dist}_c{n_components}.csv'
        )
        
        flag = 0  # Flag to indicate whether the embedding was loaded or calculated

        if os.path.exists(embedding_filename):
            # Loading the pre-calculated embedding
            print(f"Loading embedding from {embedding_filename}...")
            try:
                clusterable_embedding_df = pd.read_csv(embedding_filename)
                clusterable_embedding = clusterable_embedding_df.drop('id_molecola', axis=1).values
                flag = 1
                print(f"Loading embedding from {embedding_filename} completed.")
            except Exception as e:
                logging.error(f"Error loading embedding file {embedding_filename}: {e}")
                return {
                    'params': ",".join(map(str, params)),
                    'n_clusters_found': np.nan,
                    'silhouette': np.nan,
                    'dbcv': np.nan,
                    'embedding_file': None,
                    'label_file': None,
                    'error': f'Error loading embedding: {e}'
                }
        else:
            # Calculating the UMAP embedding
            print(f"Embedding file not found: {embedding_filename}. Calculating UMAP...")
            if not os.path.exists(fingerprints_file):
                logging.error(f"Fingerprints file not found: {fingerprints_file}")
                return {
                    'params': ",".join(map(str, params)),
                    'n_clusters_found': np.nan,
                    'silhouette': np.nan,
                    'dbcv': np.nan,
                    'embedding_file': None,
                    'label_file': None,
                    'error': 'Fingerprints file not found'
                }
            try:
                # Loading the fingerprints
                data = pd.read_csv(fingerprints_file)
                if 'id_molecola' not in data.columns:
                    logging.error("The 'id_molecola' column is not present in the fingerprints file.")
                    return {
                        'params': ",".join(map(str, params)),
                        'n_clusters_found': np.nan,
                        'silhouette': np.nan,
                        'dbcv': np.nan,
                        'embedding_file': None,
                        'label_file': None,
                        'error': "Missing 'id_molecola' column in fingerprints."
                    }
                
                X = data.drop('id_molecola', axis=1)
                id_molecole_from_fingerprints = data['id_molecola']
                
                # Check that the molecule IDs match
                if not id_molecoles.equals(id_molecole_from_fingerprints):
                    logging.warning("The molecule IDs do not match between 'molecule_ids.csv' and 'fingerprints_file'.")
                
                # Data standardization
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                logging.info("Data standardization completed.")

                # UMAP configuration
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=n_components,
                    random_state=42,
                    metric='cosine'
                )

                # Applying UMAP
                clusterable_embedding = reducer.fit_transform(X_scaled).astype(np.float64)
                logging.info("UMAP calculation completed.")

                # Creating the DataFrame with the embedding and molecule IDs
                embedding_columns = [f"UMAP_{i+1}" for i in range(clusterable_embedding.shape[1])]
                clusterable_embedding_df = pd.DataFrame(clusterable_embedding, columns=embedding_columns)
                clusterable_embedding_df['id_molecola'] = id_molecoles

                # Saving the embedding
                clusterable_embedding_df.to_csv(embedding_filename, index=False)
                logging.info(f"Saved embedding to '{embedding_filename}'.")

            except Exception as e:
                logging.error(f"Error calculating UMAP for parameters {params}: {e}")
                return {
                    'params': ",".join(map(str, params)),
                    'n_clusters_found': np.nan,
                    'silhouette': np.nan,
                    'dbcv': np.nan,
                    'embedding_file': None,
                    'label_file': None,
                    'error': f'Error calculating UMAP: {e}'
                }

        # Applying KMeans
        try:
            logging.info(f"Applying KMeans with n_clusters={n_clusters}...")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            labels = clusterer.fit_predict(clusterable_embedding)
        except Exception as e:
            logging.error(f"Error applying KMeans for parameters {params}: {e}")
            return {
                'params': ",".join(map(str, params)),
                'n_clusters_found': np.nan,
                'silhouette': np.nan,
                'dbcv': np.nan,
                'embedding_file': embedding_filename,
                'label_file': None,
                'error': f'Error in KMeans: {e}'
            }

        # Calculating the number of clusters actually found
        unique_labels = set(labels)
        num_clusters_found = len(unique_labels)

        # Calculating the silhouette score
        if num_clusters_found > 1:
            try:
                silhouette = silhouette_score(clusterable_embedding, labels)
            except Exception as e:
                logging.error(f"Error calculating silhouette for parameters {params}: {e}")
                silhouette = np.nan
        else:
            silhouette = np.nan

        # Calculating the DBCV index
        if num_clusters_found > 1:
            try:
                dbcv = validity.validity_index(clusterable_embedding, labels)
            except AttributeError as ae:
                logging.error(f"Error calculating DBCV for parameters {params}: {ae}")
                dbcv = np.nan
            except Exception as e:
                logging.error(f"General error calculating DBCV for parameters {params}: {e}")
                dbcv = np.nan
        else:
            logging.warning(f"Number of clusters <= 1 for parameters {params}, unable to calculate DBCV.")
            dbcv = np.nan

        # Saving the labels in a CSV
        try:
            labels_df = pd.DataFrame({
                'id_molecola': id_molecoles,
                'cluster': labels
            })
            label_filename = os.path.join(
                label_output_dir, 
                f'labels_n{n_neighbors}_d{min_dist}_c{n_components}_k{n_clusters}.csv'
            )
            labels_df.to_csv(label_filename, index=False)
            logging.info(f"Saved labels to '{label_filename}'.")
        except Exception as e:
            logging.error(f"Error saving labels for parameters {params}: {e}")
            label_filename = None

        # Preparing the results
        result = {
            'params': ",".join(map(str, [
                n_neighbors, min_dist, n_components, n_clusters
            ])),
            'n_clusters_found': num_clusters_found,
            'silhouette': silhouette,
            'dbcv': dbcv,
            'embedding_file': embedding_filename,
            'label_file': label_filename
        }

        logging.info(f"Completed: {result['params']}")
        return result
    except Exception as e:
        logging.error(f"General error processing parameters {params}: {e}")
        return {
            'params': ",".join(map(str, params)),
            'n_clusters_found': np.nan,
            'silhouette': np.nan,
            'dbcv': np.nan,
            'embedding_file': None,
            'label_file': None,
            'error': f'General error: {e}'
        }

def get_max_workers():
    """
    Automatically determines the maximum number of parallel processes based on physical cores.
    
    Returns:
        int: Number of parallel processes.
    """
    physical_cores = psutil.cpu_count(logical=False)
    if physical_cores:
        max_workers = physical_cores - 2
    else:
        max_workers = os.cpu_count() or 1  # Fallback to 1 if os.cpu_count() returns None
    return max_workers

def generate_unique_filename(filename):
    """
    Generates a unique filename if the file already exists.
    
    Args:
        filename (str): Filename to check.
    
    Returns:
        str: Unique filename.
    """
    base, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1
    return new_filename

def main(
    embeddings_dir='../results/umap_kmeans_validation_embeddings', 
    fingerprints_file='../results/molecule_fingerprints2.csv',
    results_file='../results/umap_kmeans_validation_results.csv', 
    label_output_dir="../results/umap_kmeans_validation_labels"
):
    """
    Main function that coordinates the clustering process using pre-calculated or on-the-fly embeddings.
    
    Args:
        embeddings_dir (str): Directory containing the pre-calculated embeddings.
        fingerprints_file (str): Path to the CSV file containing the molecule fingerprints.
        results_file (str): Name of the CSV file to save the results.
        label_output_dir (str): Directory to save the labels.
    """
    setup_logging()
    logging.info("Starting the clustering process using pre-calculated or on-the-fly embeddings.")

    # Definition of parameter lists
    n_neighbors_range = [10, 15, 20, 25, 30, 35, 40, 45]
    min_dist_range = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 0.9]
    n_components_range = [20, 25, 30, 35, 40, 45, 50]
    n_clusters_list = [10,  20, 30,  40,  50, 60,  70,  80,  90, 100,125, 150, 175, 200, 250,300]

    # Generating all parameter combinations
    param_combinations = list(itertools.product(
        n_neighbors_range, 
        min_dist_range, 
        n_components_range, 
        n_clusters_list
    ))
    logging.info(f"Total number of parameter combinations: {len(param_combinations)}")

    # remove to run the full version
    # param_combinations = param_combinations[:2]

    # Loading the molecule IDs
    molecule_ids_file = fingerprints_file
    if not os.path.exists(molecule_ids_file):
        logging.error(f"The molecule IDs file '{molecule_ids_file}' does not exist.")
        return

    if not os.path.exists(embeddings_dir):
        logging.error(f"Embeddings directory '{embeddings_dir}' doesn't exist. Creating it...")
        # create the directory
        os.makedirs(embeddings_dir, exist_ok=True)
        logging.info(f"Directory '{embeddings_dir}' created.")

    try:
        id_molecoles = pd.read_csv(molecule_ids_file)['id_molecola']
        logging.info(f"Molecule IDs loaded: {len(id_molecoles)} molecules.")
    except Exception as e:
        logging.error(f"Error loading molecule IDs from '{molecule_ids_file}': {e}")
        return

    # Preparing the directory to save the labels
    os.makedirs(label_output_dir, exist_ok=True)
    logging.info(f"Output directory for labels prepared: '{label_output_dir}'.")

    # Determining the maximum number of workers
    max_workers = get_max_workers()
    logging.info(f"Number of parallel processes set to: {max_workers}.")

    # Preparing to collect results
    results = []

    # Initializing the progress bar
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Creating futures by passing each parameter combination
        futures = {
            executor.submit(process_param, params, embeddings_dir, fingerprints_file, id_molecoles, label_output_dir): params
            for params in param_combinations
        }

        # Iterating through the futures with a progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc='Processing parameters'):
            result = future.result()
            results.append(result)

    # Creating a DataFrame of the results
    results_df = pd.DataFrame(results)
    logging.info("All parameter sets have been processed.")

    # Generate a unique filename if the file already exists
    results_file = generate_unique_filename(results_file)

    # Saving the results in a CSV
    try:
        results_df.to_csv(results_file, index=False)
        logging.info(f"Results saved in '{results_file}'.")
    except Exception as e:
        logging.error(f"Error saving results in '{results_file}': {e}")

    logging.info("Clustering process completed.")

    return results_df


def UMAP_KMeans_tuning(embeddings_dir, fp_file, results_file, label_output_dir):
    """
    Function to execute clustering with KMeans using pre-calculated or on-the-fly embeddings.

    Args:
        embeddings_dir (str): Directory containing the pre-calculated embeddings.
        fingerprints_file (str): Path to the CSV file containing the molecule fingerprints.
        results_file (str): Name of the CSV file to save the results.
        label_output_dir (str): Directory to save the labels.

    Returns:
        pandas.DataFrame: DataFrame containing the clustering results.
    """
    result_df = main(
        embeddings_dir=embeddings_dir, 
        fingerprints_file=fp_file,
        results_file=results_file, 
        label_output_dir=label_output_dir
    )

    return result_df
    
# if __name__ == '__main__':
#     UMAP_KMeans_tuning('new_molecule_fingerprints2.csv')
