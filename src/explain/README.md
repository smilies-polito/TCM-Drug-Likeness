# Fragment attribution (EdgeSHAPer)

Explanation step of the pipeline. Given the trained classifier and the TM-MC 2.0 compounds, it
assigns a Shapley value to every edge of each molecule's junction tree and exports the fragments
above threshold in the format the downstream clustering expects.

The Shapley values are computed by the authors' own implementation of EdgeSHAPer
(Mastropietro et al., *iScience* 25:105043, 2022), not by a reimplementation. The container
installs it from the authors' repository at a pinned commit; see
`src/reproducibility/containers/TCM.def`.

## Entry point

`edgeshaper_step.py` is what the pipeline calls. `run_edgeshaper_step(csv, pred_path, out_dir)`
runs the attribution, attaches the fragment SMILES to each edge, applies the retention
threshold, and returns the path of a CSV with the columns `crea_fingerprint` requires
(`smiles`, `num_neigh`, `id_molecola`, `pred`). The format matches the fragment file that
`predict_drug` used to produce from attention weights, so nothing downstream changes.

## Files

| file | role |
|---|---|
| `edgeshaper_step.py` | pipeline entry point: attribution to selected fragments |
| `edgeshaper_run.py` | adapter that exposes the model to the EdgeSHAPer package, plus the per-molecule loop |
| `explain_core.py` | model and dataset loading, batching, tree helpers, fragment export |
| `edgeshaper_export.py` | re-export fragments at a different threshold without recomputing |
| `downstream_fingerprint.py` | fragment-occurrence fingerprints |
| `cluster_published_params.py` | UMAP + HDBSCAN with the published parameters, and medoids |
| `medoids_from_clusters.py` | medoid list with cluster size and SMILES |
| `medoids_to_tcm.py` | joins the medoids to their TM-MC entries |
| `edgeshaper_statistics.py` | the figures quoted in the manuscript |
| `plot_edgeshaper_molecules.py` | manuscript figure: junction tree coloured by Shapley value |
| `plot_umap_clusters.py`, `plot_style.py` | manuscript figure: the cluster map, and shared style |

## Output

`results/explain/` holds the intermediates; `results/explain/tcm_edgeshaper/` holds what the
literature screen consumes and produces, including `medoids_tcm.csv`.

## Not here

Other attribution methods tried during development (attention, clique and pair occlusion,
integrated gradients, an independent implementation of edge Shapley values) and the scripts that
compared and diagnosed them are kept in `archive/`, which is outside version control. They are
not needed to reproduce the published results.
