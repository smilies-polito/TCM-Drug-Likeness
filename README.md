# Druglikeness prediction and Fragment Extraction Using Transformer-based Graph Neural Networks on Traditional Chinese Medicine molecules

## Project Overview

Drug-likeness is often treated as an intrinsic molecular property, yet its operational meaning depends on the chemical space and labels used to define it. We reframe drug-likeness as a dataset-defined classification problem and apply the resulting pipeline to the prioritisation of Traditional Chinese Medicine (TCM) compounds. A dual-pathway graph neural network, combining a Transformer-based graph-convolution encoder over the raw atom–bond graph with a parallel encoder over a junction-tree decomposition, was trained on a balanced classification task built from the in vivo and in vitro subsets of ZINC15 (258,500 compounds). On a held-out test set of 27,000 compounds the classifier reached a ROC-AUC of 0.931 and an F1 score of 0.797, while three published drug-likeness predictors separated the same classes no better than chance when evaluated on this data (ROC-AUC 0.37–0.53); in the reverse direction, the model trained here retained ROC-AUC of 0.75–0.84 on the external test sets associated with those predictors. Applied to 20,973 processable TM-MC 2.0 compounds, the model assigned 9,695 (46.23%) to the positive class. Fragment-level attribution was then obtained with EdgeSHAPer, a Shapley-value method applied here to the edges of the junction-tree decomposition, so that each score quantifies the contribution of a pair of adjacent cliques rather than of a single bond; retaining the fragments whose Shapley value exceeds 0.005 attributed 94,026 fragment occurrences across 862 unique fragment types, dominated by glycosidic and terpenoid/steroid substructures consistent with the source chemistry. Fragment-occurrence fingerprints, reduced with UMAP and clustered with HDBSCAN, resolved 110 chemically coherent structural families (silhouette 0.78, DBCV 0.73). A literature-reviewing agent screened the 110 family representatives: 27.3% carried documented activity for the purified compound, a further 48.2% were supported by a verified parent or analogue, and toxicity was flagged for 19; three of them, two triterpenoids and α-tocopherol, are examined as testable structural hypotheses. The pipeline is offered as a reproducible strategy for exploring TCM chemical space rather than as a universal definition of drug-likeness or a substitute for experimental validation.

## Requirements

- Linux x86_64.
- Either SingularityCE (Option A, recommended) or Python 3.10 (Option B).
- An NVIDIA GPU is recommended, with a driver supporting CUDA 12.1 or later (the PyTorch build
  used is compiled for CUDA 12.1). The comparison with published predictors requires a GPU,
  because the D-GCAN code runs only on CUDA.
- About 6.7 GB of disk space for the container image, and about 6 GB for the output of a full run
  (`results/`, roughly 190,000 files, mostly featurisation caches and per-molecule images).

### Reference environment

The published results were produced on an NVIDIA RTX A4000 (16 GB) with SingularityCE 3.10.3.
The Singularity container is the reference environment: on that machine, a container built from
`src/reproducibility/containers/TCM.def` reproduces the results reported in the manuscript byte
for byte.

## Repository structure

```
chinese-medicine-drugs/
│
├── data/                                           input data (see "Data" below)
│   │
│   ├── zinc/
│   │   ├── ZINC_dataset.csv                        258,500 compounds with the Drug label
│   │   └── splits/                                 train / validation / test sets
│   │
│   └── tm-mc2/
│       ├── preprocessed_TMMC_compounds.csv         20,973 TM-MC 2.0 compounds
│       ├── chemical_property.xlsx                  compound IDs, InChIKeys and SMILES
│       ├── medicinal_compound.xlsx                 compound names
│       └── chemical_protein.xlsx                   protein targets of each compound
│
├── results/                                        outputs of the pipelines, except:
│   └── trained_model/                              the published model and its
│                                                   clustering parameters
│
└── src/
    │
    ├── gnn/                                        the model
    │   ├── model_architecture/                     dual-pathway graph neural network
    │   ├── model.py
    │   ├── dataset_featurizer.py                   molecular graphs and junction trees
    │   ├── train.py                                training
    │   ├── validation_train.py                     hyperparameter search
    │   ├── predict.py                              prediction
    │   └── config.py
    │
    ├── explain/                                    fragment attribution (EdgeSHAPer)
    │   ├── edgeshaper_step.py                      entry point used by the pipelines
    │   ├── edgeshaper_run.py                       per-molecule loop over the tree edges
    │   ├── explain_core.py                         model, dataset and fragment export
    │   ├── edgeshaper_export.py                    re-export at a different threshold
    │   ├── downstream_fingerprint.py               fingerprints of the retained fragments
    │   ├── cluster_published_params.py             clustering with the published parameters
    │   ├── medoids_from_clusters.py                family medoids
    │   ├── medoids_to_tcm.py                       medoid annotation with TM-MC 2.0
    │   ├── edgeshaper_statistics.py                fragment summary and manuscript numbers
    │   └── plot_*.py                               manuscript figures and shared style
    │
    ├── analysis/                                   fragments, clustering, annotation
    │   ├── post_proc_pipeline.py                   fragment-occurrence fingerprints
    │   ├── PCA_*.py, UMAP_*.py                     clustering parameter search
    │   ├── apply_clustering_method.py              UMAP + HDBSCAN, medoid extraction
    │   ├── smiles_conversion.py                    SMILES to InChIKey
    │   ├── generate_centroids_with_inchikey.py     InChIKeys of the medoids
    │   └── Centroids_to_TMMC.py                    medoid annotation with TM-MC 2.0
    │
    ├── paper/
    │   ├── figures_and_statistics.py               regenerate both from results/
    │   ├── make_figures.py                         manuscript figures
    │   └── manuscript_statistics.py                every number quoted in the manuscript
    │
    ├── preprocessing/
    │   ├── zinc/                                   construction of the ZINC dataset
    │   │                                           (documentation only, see "Data")
    │   └── utils/
    │       ├── train_vali_test_division.py         train / validation / test split
    │       ├── delete_problems_molecules.py        molecule filtering
    │       └── remove_duplicate.py                 removal of shared molecules
    │
    ├── pipelines/
    │   ├── benchmark.py                            apply the published model to any dataset
    │   └── full_pipeline.py                        retrain everything from scratch
    │
    ├── reproducibility/
    │   ├── results_reproduction.py                 reproduce the published results
    │   │
    │   ├── benchmark/                              reproduce the comparison with
    │   │   │                                       published predictors
    │   │   ├── run_benchmark.sh
    │   │   ├── prepare_inputs.py
    │   │   └── build_comparison_tables.py
    │   │
    │   └── containers/                             Singularity definitions and
    │       │                                       pinned package versions
    │       ├── TCM.def                             our model
    │       ├── requirements.lock
    │       ├── druglikeness.def                    published predictors (benchmark)
    │       └── druglikeness.lock
    │
    └── literature_skill/
        └── tcm-compound-pharmacology/              Claude skill for the literature screening
```

## Installation

Pick one of the two options below and follow it end-to-end.

### Option A: Singularity container (recommended)

#### Install Singularity
Install SingularityCE following the
[installation guide](https://docs.sylabs.io/guides/3.10/admin-guide/installation.html). The
container was built and tested with SingularityCE 3.10.3.

#### Clone the repository
```bash
git clone da inserire
```

#### Build the container
Run the build from `src/reproducibility/containers/`, either with `--fakeroot`:
```bash
cd chinese-medicine-drugs/src/reproducibility/containers
singularity build --fakeroot TCM.sif TCM.def
```
or with root privileges:
```bash
cd chinese-medicine-drugs/src/reproducibility/containers
sudo singularity build TCM.sif TCM.def
```

### Option B: Local installation

Requires Python 3.10 and the system libraries `libcairo2` and `libxrender1` (on Debian/Ubuntu:
`sudo apt install libcairo2 libxrender1`).
```bash
git clone da inserire
cd chinese-medicine-drugs
python3.10 -m venv .venv
source .venv/bin/activate
pip install pip==26.2.1 setuptools==84.0.0 wheel==0.48.0
pip install -r src/reproducibility/containers/requirements.lock --extra-index-url https://download.pytorch.org/whl/cu121
```

This installs the same Python package versions as the container. 

## Usage

All commands below assume you are in the `chinese-medicine-drugs` root folder and use the
container. With a local installation, drop the `singularity exec --nv src/reproducibility/containers/TCM.sif` prefix.

### Reproduce the published results

```bash
singularity exec --nv src/reproducibility/containers/TCM.sif python3 src/reproducibility/results_reproduction.py
```

As a shortcut, the container's default runscript runs the same script:
```bash
singularity run --nv src/reproducibility/containers/TCM.sif
```

Using the pretrained model shipped in `results/trained_model/`, the script:
1. evaluates the model on the ZINC15 test set (27,000 compounds);
2. applies it to the 20,973 processable TM-MC 2.0 compounds (`data/tm-mc2/`), then assigns a
   Shapley value to every junction-tree edge with EdgeSHAPer and retains the fragments scoring
   above 0.005, for the compounds classified as positive;
3. clusters those compounds by fragment occurrence (UMAP + HDBSCAN), reusing the parameters saved
   in `results/trained_model/best_valid_clustering_params.txt`, and matches the family medoids to the
   TM-MC 2.0 database;
4. generates the manuscript figures and recomputes every number quoted in the manuscript.

The EdgeSHAPer attribution dominates the cost, because it needs 2 x M = 200 forward passes per
junction-tree edge: about 25 s per compound on an otherwise idle RTX A4000, so roughly 150 h for
the whole TM-MC 2.0 set. Every other step together takes about 1 h 40 min (about 55 min for the
ZINC15 evaluation, 45 min for the TM-MC 2.0 predictions and a few minutes for the rest), and
longer on a shared GPU. The attribution must be run as a single sequential pass over the whole
set, which is what this script does. It is a Monte Carlo estimate: the coalition masks are drawn
from a generator seeded per molecule, but the permutation of the edges comes from the global
PyTorch generator, which this script seeds once at start-up and which then advances from one
molecule to the next. The estimates therefore depend on how many compounds were processed
before, and splitting the step into shards, or running a subset, changes them. The effect is not
negligible: at M = 100 the sampling noise is of the same order as the retention threshold of
0.005, so a fragment whose value lies near the threshold can fall on either side of it. With
`--skip-zinc`
step 1 is skipped, so the classification metrics and the reference distribution in the
score-distribution figure are not produced.

Main outputs, in `results/`:

| Path | Content |
|---|---|
| `test_data/predictions_raw.csv`, `test_data/final_test_metrics.json` | predictions and metrics on the ZINC15 test set |
| `test_data/TCMBANK/predictions_raw.csv` | predicted probability and class of each TM-MC 2.0 compound |
| `explain/edgeshaper_edges.csv` | Shapley value of every junction-tree edge of every compound |
| `explain/edgeshaper_fragments.csv` | the fragments retained above the threshold, one row per edge |
| `high_attention_fragments_summary.csv` | the retained fragments, ranked by number of occurrences |
| `new_molecule_fingerprints2.csv` | fragment-occurrence fingerprints of the compounds classified as positive |
| `final_clustering/` | UMAP embedding, HDBSCAN family labels and clustering summary |
| `centroids_with_inchikey.csv`, `inchikey_dict.csv` | family medoids, with InChIKey and compound names |
| `figures/` | manuscript figures (PDF) |
| `manuscript_statistics.json`, `manuscript_statistics.txt` | every number quoted in the manuscript, recomputed from the files above |

### Recompute only the figures and the statistics

Once `results/` has been populated, the figures and the statistics can be regenerated in a few
seconds, without running the model again:
```bash
singularity exec src/reproducibility/containers/TCM.sif python3 src/paper/figures_and_statistics.py
```

### Evaluate the published model on your own dataset

```bash
singularity exec --nv src/reproducibility/containers/TCM.sif python3 src/pipelines/benchmark.py data/zinc/splits/new2_ZINC_final_balanced_test_set.csv
```
Replace the path in the example with your own CSV file, which must use the same column names as
the example file (case-sensitive): `ZINCID`, `smiles` and, optionally, `Drug`. `ZINCID` can hold
any molecule identifier, but the column must be present.
`Drug` holds the true label (1 = *in vivo*, 0 = *in vitro*); if the column is missing, the file is
treated as unlabelled and only predictions are produced. Molecules whose SMILES contain more than
one fragment are skipped.

**Note:** `benchmark.py` writes to the same files as `results_reproduction.py` and overwrites them
(`results/test_data/predictions_raw.csv` for labelled data, `results/test_data/TCMBANK/predictions_raw.csv`
for unlabelled data). Run it on a separate copy of the repository, or re-run
`results_reproduction.py` afterwards, before regenerating the figures or the statistics.

### Retrain the model from scratch

```bash
singularity exec --nv src/reproducibility/containers/TCM.sif python3 src/pipelines/full_pipeline.py
```
This runs the hyperparameter search, trains a new model, evaluates it, and repeats the fragment
analysis and the clustering. It takes considerably longer than `results_reproduction.py` and has
not been timed for this release. The new model is registered under a new ID, printed at the end of
the run; the other scripts keep using the published model. `full_pipeline.py` does not generate the
manuscript figures or statistics.

### Interactive shell
If you prefer to work interactively inside the container instead of running one command at a time:
```bash
cd chinese-medicine-drugs
singularity shell --nv src/reproducibility/containers/TCM.sif
```

## Comparison with published drug-likeness predictors

The two benchmark tables of the manuscript compare the model with three published predictors in
both directions: the published predictors evaluated on our ZINC15 test set, and our model evaluated
on the test sets associated with those predictors. A single script regenerates everything from
scratch: it clones the other predictors at the commits used in the manuscript, downloads their
published weights, converts the test sets to each tool's input format, runs all the predictions
(retraining D-GCAN, as described below) and recomputes all the metrics with a single
implementation.

The other predictors need their own container, defined in `druglikeness.def` with pinned versions
in `druglikeness.lock`. Build it once, from the same folder as `TCM.def` (with `--fakeroot` or `sudo`, as above):
```bash
cd src/reproducibility/containers
singularity build --fakeroot druglikeness.sif druglikeness.def
cd ../../..
```
Then run the benchmark from the repository root:
```bash
src/reproducibility/benchmark/run_benchmark.sh
```
The run needs internet access (GitHub and Hugging Face), about 7 GB of disk space under
`results/benchmark/work/`, and about 2 h 30 min on an RTX A4000: roughly 15 min for D-GCAN,
20 min for GeneralDL and SpecDL-ZINC, and the rest for our model on the three test sets. Single steps can be run on their own, for example
`run_benchmark.sh dgcan tables`; the list is at the top of the script. The predictions of every
model are written to `results/benchmark/predictions/`, and the two tables, with all the metrics
and confusion matrices, to `results/benchmark/comparison_summary.csv`. On the reference machine
the run reproduces the published predictions and tables byte for byte, including the retraining
of D-GCAN.

The steps performed by the script are the following.

**Our model on the external test sets.** `benchmark.py` is run, as described above, on:
- the D-GCAN test set, `dataset/data_test.txt` from [D-GCAN](https://github.com/JinYSun/D-GCAN)
  (commit `a2b2c0c52b53d6abe100d7ebaed2d3f39ea0de0b`), converted to the input format above. Four of
  its 433 molecules have multi-fragment SMILES and are skipped, leaving 429;
- the drug-likeness test set, `data/test_set/approved+zinc_test.csv` from
  [druglikeness](https://github.com/EricCai00/druglikeness)
  (commit `016abc979a672304e78a3c167ee7bd0649c888e4`), 9,770 molecules.

**D-GCAN on our test set.** Following the instructions in its README, D-GCAN is retrained on its own
training set (`dataset/data_train.txt`) with the hyperparameters given there. As the README allows,
`dataset/data_test.txt` is used as the evaluation set during training, in place of the `bRo5.txt`
of the README example. The retrained model is applied to our ZINC15 test set, converted to
D-GCAN's input format (one `SMILES label` pair per line, separated by a space), and its scores are
thresholded at 0.15, the value used in D-GCAN's `predict.py`.

**GeneralDL and SpecDL-ZINC on our test set.** The `predict.py` script of the druglikeness repository
is run unchanged with the published weights (Hugging Face repository `ericsai/generaldl`, revision
`5997588dc571e1fa51f5a87316bc7d681c73f493`, folders `generaldl` and `specdl-zinc`) on our ZINC15 test
set, converted to a CSV file with the columns `smiles` and `label`. The scores are thresholded at
0.5, the default decision threshold, since the authors do not publish a specific one.

## Literature screening skill

The literature-grounded interpretation step of the pipeline is performed by a purpose-built
Claude skill, `tcm-compound-pharmacology`, distributed with this repository:

```
src/literature_skill/tcm-compound-pharmacology/
    SKILL.md      # instructions and workflow
    kernel.py     # query construction (PubMed queries per effect and for toxicity) and output helpers
```

Given one or more compound names, the skill searches PubMed and ChEMBL for documented
pharmacological effects, and returns for each compound a short summary per effect, the
supporting references as resolvable identifiers (PMID/DOI), a toxicity flag, and an evidence
grade (High = the purified compound itself is directly linked to an effect; Middle = indirect
evidence only, such as a structurally related compound or an extract; Low = no supporting
record retrieved; Warning = reported toxicity). Names separated by `;` within one entry are
treated as synonyms of a single compound.

### Requirements and installation

The skill runs on claude.ai and requires the **PubMed** and **ChEMBL** MCP
connectors to be enabled: its searches are issued through those connectors, and without them it
returns no evidence. A bioRxiv/medRxiv connector is optional and used only for preprints.

To install it, package the folder as a zip archive and upload it in claude.ai under
*Settings > Features > Skills*:
```bash
cd src/literature_skill
zip -r tcm-compound-pharmacology.skill tcm-compound-pharmacology/
```

Note that Claude skills do not synchronise across surfaces: this skill is written for the
claude.ai environment and is not intended to be loaded by other Claude products.

### Reproducing the reported screen

The compounds screened in this work are the medoids of the structural families recovered by the
clustering stage. After running `src/reproducibility/results_reproduction.py`, they are listed in
`results/centroids_with_inchikey.csv`, together with the compound names resolved from the TM-MC 2.0
database in `results/inchikey_dict.csv`. Passing those names to the skill reproduces the evidence
table underlying the literature-grounded interpretation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
