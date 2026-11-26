"""Statistiche numeriche citate nel manoscritto.

Ricalcola, a partire dagli artefatti gia' salvati dalla pipeline, ogni valore che compare nel
testo del paper: metriche di classificazione, applicazione cross-domain, confronto fra le
distribuzioni dei punteggi, composizione dei frammenti ad alta attenzione, esito del clustering e
coerenza chimica delle famiglie. Nessuna sezione riesegue il modello: si leggono i file prodotti
da una run di `results_reproduction.py`, cosi' i numeri coincidono con quelli della run che ha
generato gli altri risultati.

Ogni sezione e' indipendente e viene saltata con un avviso se mancano i suoi input, perche' una
statistica non deve mai invalidare una run di analisi completa.

Output in <out_dir>: `manuscript_statistics.json` (leggibile da programma) e
`manuscript_statistics.txt` (leggibile da persona).
"""
import json
import os
import random

import numpy as np
import pandas as pd

# Numero di frammenti riportati nella tabella del manoscritto.
TOP_FRAGMENTS = 5

# Seed della sezione sulla coerenza delle famiglie. Quella sezione campiona coppie di molecole,
# quindi il valore riportato nel manoscritto e' riproducibile solo a parita' di seed.
COHERENCE_SEED = 0

# Numero di coppie campionate entro una famiglia numerosa e numero di coppie casuali usate come
# riferimento: valori con cui sono stati ottenuti i numeri nel testo.
PAIRS_PER_FAMILY = 4000
RANDOM_PAIRS = 20000


def _read_csv(path, what):
    """Legge un CSV richiesto segnalando l'assenza invece di sollevare."""
    if not os.path.exists(path):
        print(f"[stats] salto {what}: manca {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[stats] salto {what}: {path} non leggibile ({exc})")
        return None


def classification_metrics(out_dir):
    """Metriche sul set di valutazione ZINC15, ricalcolate dalle predizioni salvate."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 f1_score, precision_score, recall_score, roc_auc_score)

    df = _read_csv(f'{out_dir}/test_data/predictions_raw.csv', 'metriche di classificazione')
    if df is None or not {'label', 'probability', 'pred'}.issubset(df.columns):
        return None

    y, prob, pred = df['label'].values, df['probability'].values, df['pred'].values
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        'n': int(len(df)),
        'positives': int((y == 1).sum()),
        'negatives': int((y == 0).sum()),
        # ROC-AUC va calcolata sul punteggio continuo: passando le predizioni binarizzate si
        # otterrebbe la balanced accuracy, non l'area sotto la curva.
        'roc_auc': float(roc_auc_score(y, prob)),
        'f1': float(f1_score(y, pred)),
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred)),
        'recall': float(recall_score(y, pred)),
        'true_positives': int(tp), 'true_negatives': int(tn),
        'false_positives': int(fp), 'false_negatives': int(fn),
        'false_positive_rate': float(fp / (fp + tn)),
    }


def cross_domain(out_dir):
    """Composti TM-MC processati e quota assegnata alla classe positiva."""
    df = _read_csv(f'{out_dir}/test_data/TCMBANK/predictions_raw.csv', 'applicazione cross-domain')
    if df is None or 'pred' not in df.columns:
        return None
    positives = int((df['pred'] == 1).sum())
    return {
        'processable_compounds': int(len(df)),
        'positive_compounds': positives,
        'positive_fraction': float(positives / len(df)),
    }


def score_distributions(out_dir):
    """Confronto fra la distribuzione dei punteggi sul set TCM e su quello ZINC15."""
    from scipy.stats import ks_2samp

    tcm_df = _read_csv(f'{out_dir}/test_data/TCMBANK/predictions_raw.csv',
                       'confronto fra distribuzioni')
    zinc_df = _read_csv(f'{out_dir}/test_data/predictions_raw.csv', 'confronto fra distribuzioni')
    if tcm_df is None or zinc_df is None:
        return None

    tcm, zinc = tcm_df['probability'].values, zinc_df['probability'].values

    # Area comune sotto le due densita': stessi bin per entrambe, poi integrale del minimo.
    bins = np.linspace(0, 1, 61)
    width = bins[1] - bins[0]
    dens_tcm, _ = np.histogram(tcm, bins=bins, density=True)
    dens_zinc, _ = np.histogram(zinc, bins=bins, density=True)
    ks = ks_2samp(tcm, zinc)

    stats = {
        'tcm': {'n': int(len(tcm)), 'median': float(np.median(tcm)),
                'q1': float(np.percentile(tcm, 25)), 'q3': float(np.percentile(tcm, 75))},
        'zinc': {'n': int(len(zinc)), 'median': float(np.median(zinc)),
                 'q1': float(np.percentile(zinc, 25)), 'q3': float(np.percentile(zinc, 75))},
        'overlap_fraction': float(np.minimum(dens_tcm, dens_zinc).sum() * width),
        'ks_statistic': float(ks.statistic),
        'ks_pvalue': float(ks.pvalue),
    }
    if 'label' in zinc_df.columns:
        labels = zinc_df['label'].values
        stats['zinc']['median_in_vivo'] = float(np.median(zinc[labels == 1]))
        stats['zinc']['median_in_vitro'] = float(np.median(zinc[labels == 0]))
    return stats


def _fragment_class(smiles):
    """Classe chimica di un frammento, secondo una regola esplicita e ordinata.

    Aromatico se contiene atomi aromatici; altrimenti ossano se ha un anello a sei termini con
    esattamente un ossigeno (la firma dell'anello piranosico degli zuccheri); altrimenti
    carbociclo saturo se tutti gli anelli sono di soli carboni; altrimenti aciclico o eterociclo.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'unparsable'
    if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
        return 'aromatic'
    rings = mol.GetRingInfo().AtomRings()
    if not rings:
        return 'acyclic'
    for ring in rings:
        symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in ring]
        if len(ring) == 6 and symbols.count('O') == 1:
            return 'oxane'
    if all(mol.GetAtomWithIdx(i).GetSymbol() == 'C' for ring in rings for i in ring):
        return 'saturated_carbocycle'
    return 'other_heterocycle'


def fragment_statistics(out_dir, top_n=TOP_FRAGMENTS):
    """Composizione dei frammenti ad alta attenzione, pesata per occorrenze."""
    summary = _read_csv(f'{out_dir}/high_attention_fragments_summary.csv',
                        'statistiche sui frammenti')
    if summary is None:
        return None

    total = int(summary['occurrences'].sum())
    by_class = {}
    for row in summary.itertuples(index=False):
        entry = by_class.setdefault(_fragment_class(row.smiles), {'types': 0, 'occurrences': 0})
        entry['types'] += 1
        entry['occurrences'] += int(row.occurrences)
    for entry in by_class.values():
        entry['occurrence_fraction'] = entry['occurrences'] / total

    stats = {
        'total_occurrences': total,
        'unique_types': int(len(summary)),
        'by_class': dict(sorted(by_class.items(), key=lambda kv: -kv[1]['occurrences'])),
        'top_fragments': [
            {'rank': int(r.rank), 'smiles': r.smiles, 'occurrences': int(r.occurrences),
             'compounds': int(r.compounds), 'pct_of_compounds': float(r.pct_of_compounds)}
            for r in summary.head(top_n).itertuples(index=False)
        ],
    }

    # La mediana dei frammenti per composto si ricava dalle fingerprint, che contano le
    # occorrenze e non i soli tipi distinti presenti.
    fingerprints = _read_csv(f'{out_dir}/new_molecule_fingerprints2.csv',
                             'mediana dei frammenti per composto')
    if fingerprints is not None:
        counts = fingerprints.drop(columns=['id_molecola'], errors='ignore').values
        stats['median_fragments_per_compound'] = float(np.median(counts.sum(axis=1)))
    return stats


def clustering_statistics(out_dir):
    """Esito del clustering finale, letto dal riepilogo scritto dalla pipeline."""
    summary_path = f'{out_dir}/final_clustering/final_UMAP_HDBSCAN_summary.json'
    if not os.path.exists(summary_path):
        print(f"[stats] salto le statistiche di clustering: manca {summary_path}")
        return None
    with open(summary_path) as handle:
        summary = json.load(handle)

    stats = {
        'families': int(summary['num_clusters']),
        'clustered_pct': float(summary['clustered_pct']),
        'silhouette': float(summary['silhouette']),
        'dbcv': float(summary['dbcv']),
        'n_points': int(summary['n_points']),
        'params': summary.get('params', {}),
    }

    labels = _labels(out_dir)
    if labels is not None:
        sizes = pd.Series(labels[labels >= 0]).value_counts().values
        noise = int((labels == -1).sum())
        stats.update({
            'noise_compounds': noise,
            'noise_fraction': float(noise / len(labels)),
            'family_size_min': int(sizes.min()),
            'family_size_max': int(sizes.max()),
            'family_size_median': float(np.median(sizes)),
        })
    return stats


def _labels(out_dir):
    path = (f'{out_dir}/final_clustering/'
            'final_UMAP_HDBSCAN_labels_min_samples_20_min_cluster_size_30.csv')
    df = _read_csv(path, 'etichette delle famiglie')
    return None if df is None else df.iloc[:, 0].values


def family_coherence(out_dir, seed=COHERENCE_SEED):
    """Coerenza chimica delle famiglie: similarita' interna e scaffold condiviso.

    La similarita' entro una famiglia viene confrontata con quella di coppie estratte a caso fra
    tutti i composti positivi: e' il riferimento che dice se il raggruppamento per occorrenza di
    frammenti stia recuperando qualcosa di chimicamente reale o solo rumore.
    """
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')

    embedding = _read_csv(f'{out_dir}/final_clustering/final_UMAP_embedding.csv',
                          'coerenza delle famiglie')
    dataset = _read_csv(f'{out_dir}/dataset_with_inchikey.csv', 'coerenza delle famiglie')
    labels = _labels(out_dir)
    if embedding is None or dataset is None or labels is None:
        return None
    if 'id_molecola' not in embedding.columns:
        print("[stats] salto la coerenza delle famiglie: l'embedding non riporta id_molecola")
        return None

    # `id_molecola` e' l'indice posizionale nel dataset dei composti TM-MC; l'embedding e le
    # etichette sono allineati riga per riga.
    ids = embedding['id_molecola'].values
    smiles = dataset['smiles'].values
    if len(ids) != len(labels):
        print("[stats] salto la coerenza delle famiglie: etichette ed embedding disallineati")
        return None

    random.seed(seed)
    fingerprints, scaffolds = {}, {}
    for mol_id in ids:
        mol = Chem.MolFromSmiles(smiles[mol_id])
        if mol is None:
            continue
        fingerprints[mol_id] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
        try:
            scaffolds[mol_id] = MurckoScaffold.MurckoScaffoldSmiles(
                mol=MurckoScaffold.MakeScaffoldGeneric(mol))
        except Exception:
            scaffolds[mol_id] = ''

    similarities, scaffold_shares, per_family = [], [], []
    for family in sorted(set(labels[labels >= 0])):
        members = [i for i, lab in zip(ids, labels) if lab == family and i in fingerprints]
        if len(members) < 2:
            continue
        pairs = [(a, b) for k, a in enumerate(members) for b in members[k + 1:]]
        if len(pairs) > PAIRS_PER_FAMILY:
            pairs = random.sample(pairs, PAIRS_PER_FAMILY)
        similarity = float(np.mean([DataStructs.TanimotoSimilarity(fingerprints[a],
                                                                   fingerprints[b])
                                    for a, b in pairs]))
        counts = pd.Series([scaffolds[i] for i in members]).value_counts()
        share = float(counts.iloc[0] / len(members))
        similarities.append(similarity)
        scaffold_shares.append(share)
        per_family.append({'family': int(family), 'size': len(members),
                           'mean_tanimoto': similarity, 'top_scaffold_share': share,
                           'top_scaffold': counts.index[0]})

    known = list(fingerprints)
    baseline = [DataStructs.TanimotoSimilarity(fingerprints[a], fingerprints[b])
                for a, b in ((random.choice(known), random.choice(known))
                             for _ in range(RANDOM_PAIRS)) if a != b]
    baseline_median = float(np.median(baseline))

    return {
        'seed': seed,
        'families_evaluated': len(similarities),
        'within_family_tanimoto_median': float(np.median(similarities)),
        'random_pair_tanimoto_median': baseline_median,
        'families_above_random_median': int(sum(s > baseline_median for s in similarities)),
        'top_scaffold_share_median': float(np.median(scaffold_shares)),
        'families_above_half_scaffold': int(sum(s > 0.5 for s in scaffold_shares)),
        'families_above_80pct_scaffold': int(sum(s > 0.8 for s in scaffold_shares)),
        'most_homogeneous_families': sorted(per_family, key=lambda f: -f['top_scaffold_share'])[:5],
    }


def _render(stats):
    """Rende le statistiche in un rapporto leggibile."""
    lines = ['STATISTICHE CITATE NEL MANOSCRITTO', '=' * 60, '']

    c = stats.get('classification')
    if c:
        lines += ['Classificazione sul set di valutazione ZINC15',
                  f"   n = {c['n']:,}  ({c['positives']:,} positivi, {c['negatives']:,} negativi)",
                  f"   ROC-AUC {c['roc_auc']:.4f}   F1 {c['f1']:.4f}   "
                  f"accuratezza {c['accuracy']*100:.2f}%   bilanciata {c['balanced_accuracy']*100:.2f}%",
                  f"   precisione {c['precision']*100:.2f}%   richiamo {c['recall']*100:.2f}%",
                  f"   TP {c['true_positives']:,}  TN {c['true_negatives']:,}  "
                  f"FP {c['false_positives']:,}  FN {c['false_negatives']:,}   "
                  f"FPR {c['false_positive_rate']*100:.2f}%", '']

    d = stats.get('cross_domain')
    if d:
        lines += ['Applicazione cross-domain ai composti TM-MC 2.0',
                  f"   processati {d['processable_compounds']:,}   "
                  f"positivi {d['positive_compounds']:,} ({d['positive_fraction']*100:.2f}%)", '']

    s = stats.get('score_distributions')
    if s:
        lines += ['Distribuzioni dei punteggi',
                  f"   TM-MC : mediana {s['tcm']['median']:.4f}  IQR [{s['tcm']['q1']:.4f}, {s['tcm']['q3']:.4f}]",
                  f"   ZINC15: mediana {s['zinc']['median']:.4f}  IQR [{s['zinc']['q1']:.4f}, {s['zinc']['q3']:.4f}]"]
        if 'median_in_vivo' in s['zinc']:
            lines.append(f"           in vivo {s['zinc']['median_in_vivo']:.4f}   "
                         f"in vitro {s['zinc']['median_in_vitro']:.4f}")
        lines += [f"   sovrapposizione delle densita' {s['overlap_fraction']*100:.1f}%   "
                  f"Kolmogorov-Smirnov D={s['ks_statistic']:.4f}", '']

    f = stats.get('fragments')
    if f:
        lines += ['Frammenti ad alta attenzione',
                  f"   {f['total_occurrences']:,} occorrenze, {f['unique_types']} tipi unici"]
        if 'median_fragments_per_compound' in f:
            lines.append(f"   mediana per composto {f['median_fragments_per_compound']:.0f}")
        lines.append('   composizione per classe chimica (quota sulle occorrenze):')
        for name, entry in f['by_class'].items():
            lines.append(f"      {name:<22} {entry['types']:>4} tipi   "
                         f"{entry['occurrences']:>7,} occ   {entry['occurrence_fraction']*100:>5.1f}%")
        lines.append(f"   primi {len(f['top_fragments'])} frammenti:")
        for t in f['top_fragments']:
            lines.append(f"      {t['rank']:>2}  {t['smiles']:<28} {t['occurrences']:>7,} occ   "
                         f"{t['compounds']:>6,} composti  ({t['pct_of_compounds']:.2f}%)")
        lines.append('')

    cl = stats.get('clustering')
    if cl:
        lines += ['Clustering finale',
                  f"   {cl['families']} famiglie su {cl['n_points']:,} composti"]
        if 'noise_compounds' in cl:
            lines.append(f"   rumore {cl['noise_compounds']:,} ({cl['noise_fraction']*100:.2f}%)   "
                         f"dimensioni da {cl['family_size_min']} a {cl['family_size_max']}, "
                         f"mediana {cl['family_size_median']:.0f}")
        lines += [f"   silhouette {cl['silhouette']:.4f}   DBCV {cl['dbcv']:.4f}", '']

    fc = stats.get('family_coherence')
    if fc:
        lines += ['Coerenza chimica delle famiglie',
                  f"   Tanimoto entro le famiglie {fc['within_family_tanimoto_median']:.3f}   "
                  f"coppie casuali {fc['random_pair_tanimoto_median']:.3f}",
                  f"   famiglie sopra la mediana casuale: "
                  f"{fc['families_above_random_median']}/{fc['families_evaluated']}",
                  f"   scaffold di Bemis-Murcko condiviso: mediana "
                  f"{fc['top_scaffold_share_median']*100:.1f}%   "
                  f">50%: {fc['families_above_half_scaffold']}   "
                  f">80%: {fc['families_above_80pct_scaffold']}",
                  f"   (seed {fc['seed']}: la sezione campiona coppie, i valori sono "
                  f"riproducibili solo a parita' di seed)", '']

    return '\n'.join(lines)


def compute_all(out_dir='../results', top_fragments=TOP_FRAGMENTS):
    """Calcola tutte le statistiche e le salva in JSON e in un rapporto leggibile."""
    print("\nComputing manuscript statistics")
    print("################################# \n")

    sections = (
        ('classification', lambda: classification_metrics(out_dir)),
        ('cross_domain', lambda: cross_domain(out_dir)),
        ('score_distributions', lambda: score_distributions(out_dir)),
        ('fragments', lambda: fragment_statistics(out_dir, top_fragments)),
        ('clustering', lambda: clustering_statistics(out_dir)),
        ('family_coherence', lambda: family_coherence(out_dir)),
    )

    stats = {}
    for name, compute in sections:
        try:
            result = compute()
        except Exception as exc:  # una sezione che fallisce non deve fermare le altre
            print(f"[stats] sezione '{name}' fallita ({exc})")
            result = None
        if result is not None:
            stats[name] = result

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'manuscript_statistics.json')
    text_path = os.path.join(out_dir, 'manuscript_statistics.txt')
    with open(json_path, 'w') as handle:
        json.dump(stats, handle, indent=2)
    report = _render(stats)
    with open(text_path, 'w') as handle:
        handle.write(report + '\n')

    print(report)
    print(f"[stats] salvati {json_path} e {text_path}")
    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Recompute every number quoted in the manuscript from the saved pipeline '
                    'artefacts, without re-running the model.')
    parser.add_argument('--out-dir', default='../results',
                        help="directory degli artefatti della pipeline (default: ../results)")
    parser.add_argument('--top-fragments', type=int, default=TOP_FRAGMENTS,
                        help='numero di frammenti riportati nella tabella del manoscritto')
    args = parser.parse_args()
    compute_all(out_dir=args.out_dir, top_fragments=args.top_fragments)
