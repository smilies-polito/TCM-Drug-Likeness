try:
    from PCA_plots import pca_scree_plot
    from PCA_validation import PCA_param_tuning
    from UMAP_Kmeans_validation import UMAP_KMeans_tuning
    from UMAP_validation import UMAP_HDBSCAN_tuning
except:
    from PCA_plots import pca_scree_plot
    from PCA_validation import PCA_param_tuning
    from UMAP_Kmeans_validation import UMAP_KMeans_tuning
    from UMAP_validation import UMAP_HDBSCAN_tuning
import os
import pandas as pd
from apply_clustering_method import centroids_extraction  
from apply_clustering_method import apply_clustering

def crea_fingerprint(input_file, output_file, frammento_to_index=None):
    # 1. Read the input file
    df = pd.read_csv(input_file)

    # Check that the necessary columns are present
    colonne_necessarie = {'smiles', 'num_neigh', 'id_molecola', 'pred'}
    if not colonne_necessarie.issubset(df.columns):
        raise ValueError(f"The input file must contain the columns: {colonne_necessarie}")

    # Filter for pred=1
    df = df[df['pred'] == 1]
    if not frammento_to_index:
        # 2. Find the number of unique fragments
        frammenti_unici = df['smiles'].unique()
        num_frammenti_unici = len(frammenti_unici)
        print(f"Number of unique fragments: {num_frammenti_unici}")

        # Create a dictionary to assign an index to each unique fragment
        frammento_to_index = {frammento: idx for idx, frammento in enumerate(frammenti_unici)}

    # 3. Create a fingerprint for each molecule
    # Use pivot_table to count the occurrences of each SMILES for each id_molecola
    fingerprint = df.pivot_table(index='id_molecola',
                                 columns='smiles',
                                 values='pred',  # Count the occurrences of each SMILES
                                 aggfunc='count',
                                 fill_value=0)

    # Rename the columns with the fragment indices
    fingerprint.columns = [frammento_to_index[frammento] for frammento in fingerprint.columns]
    fingerprint.reset_index(inplace=True)

    # 4. Create the final DataFrame
    # In this case, fingerprint already has 'id_molecola' and the fragment columns with their counts

    # Save the final DataFrame to a CSV file
    fingerprint.to_csv(output_file, index=False)
    print(f"Fingerprint created and saved in {output_file}")

    _save_fragment_summary(df, frammento_to_index, output_file)

    return frammento_to_index


def _save_fragment_summary(df, frammento_to_index, fingerprint_path):
    """Elenco aggregato dei frammenti unici, salvato accanto alla fingerprint.

    Il file dei frammenti grezzi contiene una riga per ogni occorrenza (centinaia di migliaia),
    e la corrispondenza frammento -> colonna della fingerprint vive solo dentro il dizionario
    nel file dei parametri. Qui le due informazioni vengono unite in una tabella con una riga
    per frammento, direttamente utilizzabile per l'analisi e per le tabelle del manoscritto.
    """
    n_compounds = df['id_molecola'].nunique()
    summary = (df.groupby('smiles')
                 .agg(occurrences=('smiles', 'size'),
                      compounds=('id_molecola', 'nunique'))
                 .reset_index())
    summary['fingerprint_index'] = summary['smiles'].map(frammento_to_index)
    summary['pct_of_compounds'] = (summary['compounds'] / n_compounds * 100).round(2)
    summary = summary.sort_values('occurrences', ascending=False).reset_index(drop=True)
    summary.insert(0, 'rank', summary.index + 1)
    summary = summary[['rank', 'fingerprint_index', 'smiles',
                       'occurrences', 'compounds', 'pct_of_compounds']]

    path = os.path.join(os.path.dirname(fingerprint_path) or '.',
                        'high_attention_fragments_summary.csv')
    summary.to_csv(path, index=False)
    print(f"Fragment summary ({len(summary)} unique fragments) saved in {path}")
    return path

def result_df_analysis(result_df):
    """
    Analyzes a DataFrame containing the results of UMAP + KMeans and returns the parameters with the highest average between silhouette and dbcv.

    Args:
        result_df (pd.DataFrame): DataFrame with columns 'params', 'n_clusters_found', 'silhouette', 'dbcv'.

    Returns:
        tuple: The best parameters based on the average of silhouette and dbcv.
    """
    # Filter for percent_clustered >= 80
    if 'percent_clustered' in result_df.columns:
        result_df = result_df[result_df['percent_clustered'] >= 80]

    if result_df.empty:
        print("No result meets the criterion of percent_clustered >= 80")
        return None, None, None

    # Calculate the average between silhouette and dbcv
    result_df['mean_score'] = result_df[['silhouette', 'dbcv']].mean(axis=1)
    
    # Find the index of the row with the highest average
    best_index = result_df['mean_score'].idxmax()
    
    # Extract the best parameters
    best_params = result_df.loc[best_index, 'params']
    best_silhouette = result_df.loc[best_index, 'silhouette']
    best_dbcv = result_df.loc[best_index, 'dbcv']
    
    return best_params, best_silhouette, best_dbcv


def pp_pipeline(deep_search=False, frag_file="../results/test_data/TCMBANK/high_attention_frags.csv", fp_path="../results/new_molecule_fingerprints2.csv", out_dir = "../results"):

    # 1. Create a fingerprint for each molecule

    input_file = frag_file
        
    print("Creating fingerprint...")
    print("################################# \n\n\n")
    frammento_to_index = crea_fingerprint(input_file, fp_path)
    print("################################# \n\n\n")

    # 2. Obtain the best parameters for PCA
    print("Starting PC analysis...")
    print("################################# \n\n\n")
    pc = pca_scree_plot(fp_path)
    print("Starting PCA parameter tuning...")
    print("################################# \n\n\n")
    best_PCA_kmeans_silhouette, best_PCA_kmeans_n_clusters, best_PCA_hdbscan_silhouette, best_PCA_hdbscan_min_sample, best_PCA_hdbscan_min_cluster_size = PCA_param_tuning(pc, fp_path=fp_path, out_dir=out_dir)
    print("################################# \n\n\n")

    if best_PCA_kmeans_silhouette > best_PCA_hdbscan_silhouette:
        PCA_best_params = {'n_clusters': best_PCA_kmeans_n_clusters, "n_pc": pc}
        PCA_best_silhouette = best_PCA_kmeans_silhouette
        PCA_best_cluster_method = 'KMeans'

    else:
        PCA_best_params = {
            'min_samples': best_PCA_hdbscan_min_sample,
            'min_cluster_size': best_PCA_hdbscan_min_cluster_size,
            "n_pc": pc
        }
        PCA_best_silhouette = best_PCA_hdbscan_silhouette
        PCA_best_cluster_method = 'HDBSCAN'

    print("For PCA the best parameters are: ", PCA_best_params, " with Silhouette Score: ", PCA_best_silhouette)
    print("################################# \n\n\n")
    
    # 3. Get the best parameters for UMAP + KMeans
    print("Starting UMAP + KMeans analysis...")
    print("################################# \n\n\n")
    result_UMAP_KMeans_df = UMAP_KMeans_tuning(embeddings_dir=f"{out_dir}/umap_kmeans_validation_embeddings", fp_file=fp_path, results_file=f"{out_dir}/umap_kmeans_validation_results.csv", label_output_dir=f"{out_dir}/umap_kmeans_validation_labels")

    print("Starting UMAP + HDBSCAN analysis...")
    print("################################# \n\n\n")
    UMAP_KMeans_best_params, UMAP_KMeans_best_silhouette, UMAP_KMeans_best_dbcv = result_df_analysis(result_UMAP_KMeans_df)

    # 4. Get the best parameters for UMAP + HDBSCAN
    print("Starting UMAP + HDBSCAN analysis...")
    print("################################# \n\n\n")
    result_UMAP_HDBSCAN_df = UMAP_HDBSCAN_tuning(embeddings_dir=f"{out_dir}/umap_hdbscan_validation_embeddings", fp_file=fp_path, results_file=f"{out_dir}/umap_hdbscan_validation_results.csv", label_output_dir=f"{out_dir}/umap_hdbscan_validation_labels")
    print("################################# \n\n\n")
    print("Starting UMAP + HDBSCAN analysis...")
    print("################################# \n\n\n")
    UMAP_HDBSCAN_best_params, UMAP_HDBSCAN_best_silhouette, UMAP_HDBSCAN_best_dbcv = result_df_analysis(result_UMAP_HDBSCAN_df)
    print("################################# \n\n\n")

    # 5. Compare the results of UMAP + KMeans and UMAP + HDBSCAN
    print("Comparing results...")
    print("################################# \n\n\n")
    if UMAP_KMeans_best_silhouette is None:
        print("UMAP_KMeans_best_silhouette is None")
        UMAP_KMeans_best_silhouette = -1
    if UMAP_HDBSCAN_best_silhouette is None:
        print("UMAP_HDBSCAN_best_silhouette is None")
        UMAP_HDBSCAN_best_silhouette = -1

    if UMAP_KMeans_best_dbcv is None:
        print("UMAP_KMeans_best_dbcv is None")
        UMAP_KMeans_best_dbcv = -1
    if UMAP_HDBSCAN_best_dbcv is None:
        print("UMAP_HDBSCAN_best_dbcv is None")
        UMAP_HDBSCAN_best_dbcv = -1

    UMAP_KMeans_mean = (UMAP_KMeans_best_silhouette + UMAP_KMeans_best_dbcv) / 2
    UMAP_HDBSCAN_mean = (UMAP_HDBSCAN_best_silhouette + UMAP_HDBSCAN_best_dbcv) / 2

    if UMAP_KMeans_best_silhouette != -1 and UMAP_KMeans_mean > UMAP_HDBSCAN_mean:

        best_UMAP_params = UMAP_KMeans_best_params
        params_values = best_UMAP_params.split(',')
        best_UMAP_params = {
            'n_neighbors': int(float(params_values[0])),
            'min_dist': float(params_values[1]),
            'n_components': int(float(params_values[2])),
            'n_clusters': int(float(params_values[3]))
        }
        best_UMAP_silhouette = UMAP_KMeans_best_silhouette
        best_UMAP_dbcv = UMAP_KMeans_best_dbcv
        UMAP_best_cluster_method = 'KMeans'
    else:
        if UMAP_HDBSCAN_best_silhouette != -1:
            best_UMAP_params = UMAP_HDBSCAN_best_params
            # best UMAP params is a string so we need to convert it to a dictionary with keys 'n_neighbors','min_dist','n_components','min_samples','min_cluster_size'
            params_values = best_UMAP_params.split(',')
            best_UMAP_params = {
                'n_neighbors': int(float(params_values[0])),
                'min_dist': float(params_values[1]),
                'n_components': int(float(params_values[2])),
                'min_samples': int(float(params_values[3])),
                'min_cluster_size': int(float(params_values[4]))
            }

            best_UMAP_silhouette = UMAP_HDBSCAN_best_silhouette
            best_UMAP_dbcv = UMAP_HDBSCAN_best_dbcv
            UMAP_best_cluster_method = 'HDBSCAN'
        else:
            print("Both UMAP_HDBSCAN_best_params and UMAP_KMeans_best_params are None")
            best_UMAP_params = None
            best_UMAP_silhouette = -1
            best_UMAP_dbcv = -1
            UMAP_best_cluster_method = None

    print("For UMAP the best parameters are: ", best_UMAP_params, " with Silhouette Score: ", best_UMAP_silhouette)
    print("################################# \n\n\n")
    
    # 6. Compare the results of PCA + KMeans and PCA + HDBSCAN
    if PCA_best_silhouette > best_UMAP_silhouette or best_UMAP_silhouette == -1:
        best_silhouette = PCA_best_silhouette
        best_param = PCA_best_params
        print(f"PCA is the best with parameters {best_param} and silhouette {best_silhouette}")
        return "PCA", PCA_best_cluster_method, best_param, best_silhouette, None, frammento_to_index
    else:
        best_silhouette = best_UMAP_silhouette
        best_param = best_UMAP_params
        best_dbcv = best_UMAP_dbcv
        print(f"UMAP is the best with parameters {best_param} and silhouette {best_silhouette} and dbcv {best_dbcv}")
        return "UMAP", UMAP_best_cluster_method, best_param, best_silhouette, best_dbcv, frammento_to_index
