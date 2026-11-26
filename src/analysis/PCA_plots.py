import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD


def scree_plot(X_scaled):
    # Compute the explained variance
    pca = PCA()
    pca.fit(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    pc_90 = np.argmax(np.cumsum(explained_variance) > 0.90) + 1
    print(f'Scree_plot: To explain at least 90% of the variance, at least {pc_90} principal components are needed.')
    os.makedirs('../results', exist_ok=True)
    # Generate the Scree Plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance')
    plt.title('Scree Plot')
    plt.grid()
    plt.savefig('../results/Scree_Plot.png')
    return pc_90
    

def varianza_accumulata(X_scaled):
    # Compute the explained variance
    pca = PCA()
    pca.fit(X_scaled)

    explained_variance = pca.explained_variance_ratio_

    # Compute the cumulative variance
    cumulative_variance = np.cumsum(explained_variance)

    pc_min = np.argmax(cumulative_variance > 0.90) + 1

    print(f'To explain at least 90% of the variance, at least {pc_min} principal components are needed.')
    os.makedirs('../results', exist_ok=True)
    # Generate the cumulative variance plot
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance')
    plt.title('Cumulative Variance')
    plt.grid()
    plt.savefig('../results/Varianza_Accumulata.png')
    
def pca_scree_plot(fp_path):
    # 2. Separate the features and the molecule ID
    # Assume that 'id_molecola' is a column in your CSV
    fingerprint = pd.read_csv(fp_path)
    X = fingerprint.drop('id_molecola', axis=1)
    id_molecule = fingerprint['id_molecola']

    # 3. Data Preprocessing
    # It is important to standardize the data before applying PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Apply PCA
    # Compute PCA with the maximum number of components equal to the number of features
    pca = PCA()
    pca.fit(X_scaled)

    pc_90 = scree_plot(X_scaled)


    return pc_90

# if __name__ == '__main__':
#     pca_scree_plot('postprocessing/confronto_UMAP_PCA/new_new_molecule_fingerprints2.csv')
