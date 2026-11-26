#!/usr/bin/env bash
#
# Rigenera da zero il benchmark del paper: predizioni di tutti i modelli e tabelle di confronto.
#
#   clone     clona D-GCAN e druglikeness (GeneralDL) ai commit usati nel paper e scarica i pesi
#             GeneralDL / SpecDL-ZINC / ChemBERTa da HuggingFace alla revisione usata nel paper
#   inputs    converte i test set nei formati di input di ciascun modello (prepare_inputs.py)
#   ours      nostro modello sul nostro test ZINC e sui test set di D-GCAN e druglikeness
#   dgcan     riaddestra D-GCAN sui suoi dati e predice il nostro test ZINC
#   generaldl GeneralDL e SpecDL-ZINC (pesi pubblicati) sul nostro test ZINC
#   tables    ricalcola le metriche con build_comparison_tables.py
#
# Uso (dalla radice del repo):
#        src/reproducibility/benchmark/run_benchmark.sh                 tutti i passi, in ordine
#        src/reproducibility/benchmark/run_benchmark.sh dgcan tables    solo i passi indicati
#
# Uscite:  results/benchmark/predictions/        predizioni di tutti i modelli
#          results/benchmark/comparison_summary.csv
#          results/benchmark/work/               cloni, pesi, input convertiti, log (~6 GB)
#
# Variabili d'ambiente opzionali:
#   TCM_SIF      container del nostro modello (default: src/reproducibility/containers/TCM.sif)
#   DL_SIF       container per D-GCAN e GeneralDL (default: .../containers/druglikeness.sif)
#   DGCAN_MODEL  model.pth di D-GCAN gia' addestrato: se indicato, il riaddestramento viene saltato
#
# Il nostro modello gira su una copia minima del repo (work/tcm_copy/: codice e modello
# pubblicato), perche' benchmark.py scrive sempre in results/test_data/ e sovrascriverebbe le
# predizioni usate per le figure del paper. Il modello non viene mai riaddestrato.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONTAINERS="$REPO_DIR/src/reproducibility/containers"
TCM_SIF="${TCM_SIF:-$CONTAINERS/TCM.sif}"
DL_SIF="${DL_SIF:-$CONTAINERS/druglikeness.sif}"

DGCAN_URL="https://github.com/JinYSun/D-GCAN.git"
DGCAN_COMMIT="a2b2c0c52b53d6abe100d7ebaed2d3f39ea0de0b"
DL_URL="https://github.com/EricCai00/druglikeness.git"
DL_COMMIT="016abc979a672304e78a3c167ee7bd0649c888e4"
HF_REPO="ericsai/generaldl"
HF_REVISION="5997588dc571e1fa51f5a87316bc7d681c73f493"

OUT_DIR="$REPO_DIR/results/benchmark"
PRED_DIR="$OUT_DIR/predictions"
WORK_DIR="$OUT_DIR/work"
EXT_DIR="$WORK_DIR/external"
INPUT_DIR="$WORK_DIR/inputs"
LOG_DIR="$WORK_DIR/logs"
TCM_COPY="$WORK_DIR/tcm_copy"
DGCAN_DIR="$EXT_DIR/D-GCAN"
DL_DIR="$EXT_DIR/druglikeness"
ZINC_TEST="$REPO_DIR/data/zinc/splits/new2_ZINC_final_balanced_test_set.csv"

# Isola Python dai pacchetti in ~/.local anche se DL_SIF e' un'immagine costruita senza
# PYTHONNOUSERSITE (druglikeness.def e TCM.def lo impostano gia').
export SINGULARITYENV_PYTHONNOUSERSITE=1

mkdir -p "$EXT_DIR" "$INPUT_DIR" "$PRED_DIR" "$LOG_DIR"

header() {
    echo
    echo "======================================================================"
    echo "[$(date '+%F %T')] $*"
    echo "======================================================================"
}

clone_at() {
    local url=$1 dir=$2 commit=$3
    if [ ! -d "$dir/.git" ]; then
        git clone --quiet "$url" "$dir"
    fi
    git -C "$dir" checkout --quiet "$commit"
    if [ "$(git -C "$dir" rev-parse HEAD)" != "$commit" ]; then
        echo "ERRORE: $dir non e' al commit $commit" >&2
        exit 1
    fi
    echo "$dir @ $commit"
}

step_clone() {
    header "clone: repository esterni e pesi GeneralDL"
    clone_at "$DGCAN_URL" "$DGCAN_DIR" "$DGCAN_COMMIT"
    clone_at "$DL_URL" "$DL_DIR" "$DL_COMMIT"

    # Solo le cartelle che servono (circa 3,5 GB su 8): ChemBERTa e' il backbone dei due modelli.
    singularity exec "$DL_SIF" python3 - "$DL_DIR/weights" "$EXT_DIR/.hf_cache" "$HF_REPO" "$HF_REVISION" <<'EOF'
import sys
from huggingface_hub import snapshot_download
local_dir, cache_dir, repo_id, revision = sys.argv[1:]
snapshot_download(repo_id, revision=revision, local_dir=local_dir, local_dir_use_symlinks=False,
                  cache_dir=cache_dir, allow_patterns=["chemberta/*", "generaldl/*", "specdl-zinc/*"])
print(f"pesi {repo_id}@{revision} in {local_dir}")
EOF
}

step_inputs() {
    header "inputs: conversione dei test set"
    singularity exec "$TCM_SIF" python3 "$SCRIPT_DIR/prepare_inputs.py" \
        --zinc-test "$ZINC_TEST" \
        --dgcan-test "$DGCAN_DIR/dataset/data_test.txt" \
        --druglikeness-test "$DL_DIR/data/test_set/approved+zinc_test.csv" \
        --out-dir "$INPUT_DIR"
}

run_ours() {
    local input=$1 out=$2
    rm -f "$TCM_COPY/results/test_data/predictions_raw.csv"
    (cd "$TCM_COPY" && singularity exec --nv "$TCM_SIF" python3 src/pipelines/benchmark.py "$input")
    cp "$TCM_COPY/results/test_data/predictions_raw.csv" "$PRED_DIR/$out"
    echo "-> $PRED_DIR/$out"
}

step_ours() {
    header "ours: copia minima del repo"
    # Solo quanto serve a benchmark.py per applicare il modello pubblicato: src/ (senza i .sif) e,
    # da results/trained_model/, i parametri e i pesi dell'unico run usato (vedi model_id ed
    # experiment_id in benchmark.py). Niente data/: i dati di input vengono letti dal repo, che
    # resta in sola lettura. results/processed (cache delle molecole featurizzate) sta solo nella
    # copia e, non esistendo nel repo, non viene toccata da --delete: le esecuzioni successive la
    # riusano.
    local exp=418303031868899849 run=64582bc220f74718892ff3eb63ba31a0
    local tm=results/trained_model
    mkdir -p "$TCM_COPY/$tm/mlruns/$exp" "$TCM_COPY/$tm/mlartifacts/$exp" "$TCM_COPY/results/test_data"
    rsync -a --delete --exclude __pycache__ --exclude '*.sif' "$REPO_DIR/src/" "$TCM_COPY/src/"
    rsync -a --delete "$REPO_DIR/$tm/mlruns/$exp/$run/" "$TCM_COPY/$tm/mlruns/$exp/$run/"
    rsync -a --delete "$REPO_DIR/$tm/mlartifacts/$exp/$run/" "$TCM_COPY/$tm/mlartifacts/$exp/$run/"

    header "ours: nostro modello sul nostro test ZINC"
    run_ours "$ZINC_TEST" our_on_ZINC.csv
    header "ours: nostro modello sul test set di D-GCAN"
    run_ours "$INPUT_DIR/dgcan_test.csv" our_on_DGCANtest.csv
    header "ours: nostro modello sul test set di druglikeness"
    run_ours "$INPUT_DIR/druglikeness_test.csv" our_on_druglikenesstest.csv
}

step_dgcan() {
    cp "$INPUT_DIR/zinc_test_dgcan.txt" "$DGCAN_DIR/dataset/"
    mkdir -p "$DGCAN_DIR/DGCAN/model"
    cd "$DGCAN_DIR/DGCAN"

    if [ -n "${DGCAN_MODEL:-}" ]; then
        header "dgcan: uso il modello gia' addestrato $DGCAN_MODEL"
        cp "$DGCAN_MODEL" model/model.pth
    else
        header "dgcan: addestramento sui dati di D-GCAN (iperparametri del loro README)"
        rm -f model/model.pth
        singularity exec --nv "$DL_SIF" python3 - <<'EOF'
import train
train.train('../dataset/data_test.txt', radius=1, dim=52, layer_hidden=4, layer_output=10,
            dropout=0.45, batch_train=8, batch_test=8, lr=3e-4, lr_decay=0.85, decay_interval=25,
            iteration=140, N=5000, dataset_train='../dataset/data_train.txt')
EOF
    fi

    header "dgcan: predizione sul nostro test ZINC"
    singularity exec --nv "$DL_SIF" python3 - "$PRED_DIR/DGCAN_on_ZINC.npy" <<'EOF'
import sys
import numpy as np
import predict
res = predict.predict('../dataset/zinc_test_dgcan.txt', property=True, radius=1, dim=52,
                      layer_hidden=4, layer_output=10, dropout=0.45, batch_train=8, batch_test=8,
                      lr=3e-4, lr_decay=0.85, decay_interval=25, iteration=140, N=5000)
np.save(sys.argv[1], res)
print('->', sys.argv[1], res.shape)
EOF
    cd "$WORK_DIR"
}

step_generaldl() {
    cp "$INPUT_DIR/zinc_test_generaldl.csv" "$DL_DIR/data/"
    cd "$DL_DIR"
    header "generaldl: GeneralDL sul nostro test ZINC"
    singularity exec --nv "$DL_SIF" python3 predict.py -i data/zinc_test_generaldl.csv \
        -m weights/generaldl -o "$PRED_DIR/GeneralDL_on_ZINC.csv"
    header "generaldl: SpecDL-ZINC sul nostro test ZINC"
    singularity exec --nv "$DL_SIF" python3 predict.py -i data/zinc_test_generaldl.csv \
        -m weights/specdl-zinc -o "$PRED_DIR/SpecDLZINC_on_ZINC.csv"
    # predict.py di GeneralDL scrive il suo log accanto al file di uscita: lo spostiamo nei log
    mv -f "$PRED_DIR/test_log.txt" "$LOG_DIR/generaldl_test_log.txt" 2>/dev/null || true
    cd "$WORK_DIR"
}

step_tables() {
    header "tables: metriche e tabelle di confronto"
    singularity exec "$TCM_SIF" python3 "$SCRIPT_DIR/build_comparison_tables.py"
}

ALL_STEPS=(clone inputs ours dgcan generaldl tables)
STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=("${ALL_STEPS[@]}")

for s in "${STEPS[@]}"; do
    case "$s" in
        clone|inputs|ours|dgcan|generaldl|tables) "step_$s" ;;
        *) echo "Passo sconosciuto: $s (validi: ${ALL_STEPS[*]})" >&2; exit 1 ;;
    esac
done
header "fine"
