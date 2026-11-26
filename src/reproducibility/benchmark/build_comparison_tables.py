#!/usr/bin/env python3
"""
Legge gli output grezzi delle run (nostro modello su nostri/loro dati, D-GCAN, GeneralDL e
SpecDL-ZINC sui nostri dati) e calcola tutte le metriche per tutti i modelli, usando la soglia
di decisione PROPRIA di ciascun modello dove e' stata esplicitamente pubblicata/usata dagli
autori, e 0.5 come default neutro dove non ne hanno mai definita una:

  - Nostro modello: soglia 0.5 (la nostra scelta)
  - D-GCAN:         soglia 0.15 (hardcoded nel loro predict.py, e' la soglia con cui pubblicano
                     i risultati)
  - GeneralDL / SpecDL-ZINC: non definiscono mai una soglia nel loro codice/paper (riportano
                     solo ROC-AUC) - qui usiamo 0.5 come default neutro, non una scelta loro.

Il ROC-AUC e' sempre calcolato sulle probabilita' continue, mai sulle predizioni binarizzate.
Salva anche un CSV riassuntivo con tutte le righe/metriche.

Le predizioni vengono lette da results/benchmark/predictions/ e il CSV viene scritto in
results/benchmark/comparison_summary.csv; i percorsi sono ricavati dalla posizione dello script
(src/reproducibility/benchmark/, tre livelli sotto la radice del repo), quindi
funziona da qualunque cartella venga lanciato.
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

_REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_DIR = os.path.join(_REPO_DIR, "results", "benchmark")
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
CSV_OUT = os.path.join(RESULTS_DIR, "comparison_summary.csv")


def metrics_from_scores(name, group, labels, probs, threshold, threshold_is_default):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return {
        "group": group,
        "name": name,
        "n": len(labels),
        "threshold": threshold,
        "threshold_is_default": threshold_is_default,
        "F1": f1_score(labels, preds),
        "Accuracy": accuracy_score(labels, preds),
        "Balanced Accuracy": (sensitivity + specificity) / 2,
        "Precision": precision_score(labels, preds),
        "Recall": recall_score(labels, preds),
        "ROC-AUC": roc_auc_score(labels, probs),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def load_our_csv(path):
    df = pd.read_csv(path)
    return df["label"], df["probability"]


def load_dgcan_npy(path):
    arr = np.load(path)
    return arr[:, 0], arr[:, 2]


def load_generaldl_csv(path):
    df = pd.read_csv(path)
    return df["label"], df["prediction"]


def print_table(title, rows):
    print()
    print("=" * 110)
    print(title)
    print("=" * 110)
    header = (f'{"Modello":<28}{"n":>7}{"soglia":>8}{"F1":>9}{"Accuracy":>10}'
              f'{"Bal.Acc":>9}{"Precision":>11}{"Recall":>9}{"ROC-AUC":>9}')
    print(header)
    print("-" * len(header))
    for r in rows:
        thr_str = f'{r["threshold"]:.2f}' + ('*' if r["threshold_is_default"] else '')
        line = (f'{r["name"]:<28}{r["n"]:>7}{thr_str:>8}{r["F1"]:>9.4f}{r["Accuracy"]:>10.4f}'
                f'{r["Balanced Accuracy"]:>9.4f}{r["Precision"]:>11.4f}{r["Recall"]:>9.4f}{r["ROC-AUC"]:>9.4f}')
        print(line)
        print(f'    Confusion Matrix (righe=vero, colonne=predetto): [[{r["TN"]}, {r["FP"]}], [{r["FN"]}, {r["TP"]}]]')
    print("(* soglia di default 0.5, non una scelta esplicita pubblicata dagli autori)")


def main():
    rows = []

    # 1) sul nostro campo (ZINC test, 27000)
    rows.append(metrics_from_scores("Nostro modello", "A", *load_our_csv(f"{PRED_DIR}/our_on_ZINC.csv"),
                                     threshold=0.5, threshold_is_default=False))
    rows.append(metrics_from_scores("D-GCAN (riallenato)", "A", *load_dgcan_npy(f"{PRED_DIR}/DGCAN_on_ZINC.npy"),
                                     threshold=0.15, threshold_is_default=False))
    rows.append(metrics_from_scores("GeneralDL", "A", *load_generaldl_csv(f"{PRED_DIR}/GeneralDL_on_ZINC.csv"),
                                     threshold=0.5, threshold_is_default=True))
    rows.append(metrics_from_scores("SpecDL-ZINC", "A", *load_generaldl_csv(f"{PRED_DIR}/SpecDLZINC_on_ZINC.csv"),
                                     threshold=0.5, threshold_is_default=True))
    print_table("TABELLA A - Sul nostro campo (ZINC test, 27000 molecole)",
                [r for r in rows if r["group"] == "A"])

    # 2) sul loro campo (nostro modello fuori dominio)
    rows.append(metrics_from_scores("Nostro su test D-GCAN", "B", *load_our_csv(f"{PRED_DIR}/our_on_DGCANtest.csv"),
                                     threshold=0.5, threshold_is_default=False))
    rows.append(metrics_from_scores("Nostro su test druglikeness", "B", *load_our_csv(f"{PRED_DIR}/our_on_druglikenesstest.csv"),
                                     threshold=0.5, threshold_is_default=False))
    print_table("TABELLA B - Sul loro campo (nostro modello su test set esterni)",
                [r for r in rows if r["group"] == "B"])

    print()
    print("Nota: soglia = quella pubblicata/usata dagli autori del modello (nostro=0.5, D-GCAN=0.15).")
    print("Per GeneralDL/SpecDL-ZINC, che non ne definiscono mai una nel loro codice/paper (riportano")
    print("solo ROC-AUC), e' stato usato 0.5 come default neutro (segnato con *), non una loro scelta.")
    print("Bal.Acc (balanced accuracy) = media di sensibilita' e specificita' alla soglia indicata:")
    print("misura di correttezza piu' onesta della semplice accuracy quando le predizioni sono")
    print("sbilanciate verso una classe. ROC-AUC sempre dalle probabilita' continue, mai binarizzate.")

    df_out = pd.DataFrame(rows)[["group", "name", "n", "threshold", "threshold_is_default",
                                  "F1", "Accuracy", "Balanced Accuracy", "Precision", "Recall",
                                  "ROC-AUC", "TN", "FP", "FN", "TP"]]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df_out.to_csv(CSV_OUT, index=False)
    print(f"\nCSV riassuntivo salvato in {CSV_OUT}")


if __name__ == "__main__":
    main()
