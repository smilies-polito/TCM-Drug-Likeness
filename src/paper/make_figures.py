"""Generazione delle figure per il manoscritto.

Prodotte a partire dagli artefatti salvati dalla pipeline (probabilita' continue, embedding
ridotto, etichette dei cluster, medoidi, immagini per-molecola con l'attenzione). Ogni figura
viene saltata con un avviso se mancano i suoi input, cosi' l'esecuzione della pipeline non si
interrompe per un problema di sola visualizzazione.

Output in <out_dir>/figures, in PDF (formato atteso dal manoscritto).
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

THRESHOLD = 0.5


def _figures_dir(out_dir):
    path = os.path.join(out_dir, 'figures')
    os.makedirs(path, exist_ok=True)
    return path


def tcm_score_distribution(out_dir, tcm_predictions_path, reference_predictions_path=None):
    """Distribuzione dei punteggi predetti sul set TCM, sovrapposta a quella di riferimento
    ottenuta sul test set ZINC. Mostra quanto i composti TCM cadano fuori dallo spazio
    chimico su cui il modello e' stato addestrato."""
    if not os.path.exists(tcm_predictions_path):
        print(f"[figures] salto tcm_score_distribution: manca {tcm_predictions_path}")
        return None

    tcm = pd.read_csv(tcm_predictions_path)['probability'].values

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 61)
    ax.hist(tcm, bins=bins, density=True, alpha=0.6, label=f'TM-MC 2.0 (n={len(tcm):,})')

    if reference_predictions_path and os.path.exists(reference_predictions_path):
        ref = pd.read_csv(reference_predictions_path)
        col = 'probability' if 'probability' in ref.columns else ref.columns[-1]
        ref = ref[col].values
        ax.hist(ref, bins=bins, density=True, alpha=0.6,
                label=f'ZINC15 test set (n={len(ref):,})')
    else:
        print(f"[figures] distribuzione di riferimento non trovata, grafico con il solo set TCM")

    ax.axvline(THRESHOLD, color='k', linestyle='--', linewidth=1,
               label=f'decision threshold ({THRESHOLD})')
    ax.set_xlabel('Predicted positive-class probability')
    ax.set_ylabel('Density')
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()

    path = os.path.join(_figures_dir(out_dir), 'tcm_score_distribution.pdf')
    fig.savefig(path)
    plt.close(fig)
    print(f"[figures] salvata {path}")
    return path


def umap_clusters(out_dir, embedding_path, labels_path, centroids_path=None, zoom_pct=2.0):
    """Famiglie strutturali sulle prime due componenti UMAP, in un pannello unico.

    Si usano direttamente `umap1` e `umap2` dell'embedding su cui e' stato fatto il clustering,
    senza ulteriori riduzioni. Le alternative erano peggiori: una PCA a 2D conserva solo ~36%
    della varianza e collassa tutto in un grumo, mentre ricalcolare una UMAP a 2 componenti
    spezza le famiglie in piu' frammenti scollegati (verificato: 10 su 10 fra le maggiori).

    La finestra si ricava dalle posizioni delle famiglie, non dai singoli punti. Il rumore si
    allontana molto (su EdgeSHAPer umap1 va da -5 a 22) e dilaterebbe gli assi fino a
    schiacciare le famiglie in un grumo; inoltre ogni criterio piazza una o due famiglie molto
    lontano dalle altre (sull'attenzione a umap1 -1.2 e 23.9, contro 8.4-9.3 di tutte le
    altre), e bastano quelle a svuotare la figura. Si scartano percio' le famiglie la cui
    mediana cade fuori dai percentili `zoom_pct` delle mediane, e si inquadra l'estensione
    delle rimanenti; il titolo dichiara quante famiglie e quanti composti restano in vista.
    Con `zoom_pct=0` non si scarta nulla e la vista contiene tutte le famiglie.

    I medoidi non vengono piu' cerchiati: i cerchi si sovrapponevano ai punti delle famiglie
    senza aggiungere informazione, rendendo la lettura piu' confusa. `centroids_path` resta
    accettato per compatibilita' con i chiamanti, ma non viene usato.
    """
    for p in (embedding_path, labels_path):
        if not os.path.exists(p):
            print(f"[figures] salto umap_clusters: manca {p}")
            return None

    emb = pd.read_csv(embedding_path)
    labels = pd.read_csv(labels_path).iloc[:, 0].values

    if len(labels) != len(emb):
        print(f"[figures] salto umap_clusters: {len(labels)} etichette per {len(emb)} punti")
        return None
    if not {'umap1', 'umap2'}.issubset(emb.columns):
        print("[figures] salto umap_clusters: l'embedding non ha colonne umap1/umap2")
        return None

    x, y = emb['umap1'].values, emb['umap2'].values
    noise = labels == -1
    families = sorted(set(labels[~noise]))
    sizes = pd.Series(labels[~noise]).value_counts()
    # una palette categorica non basta per ~100 famiglie: si campiona una mappa continua
    colors = dict(zip(families,
                      plt.get_cmap('gist_ncar')(np.linspace(0, 0.95, len(families)))))

    p = min(max(float(zoom_pct), 0.0), 49.0)
    centres = pd.DataFrame({'x': x[~noise], 'y': y[~noise],
                            'fam': labels[~noise]}).groupby('fam').median()
    kept = centres[(centres['x'].between(np.percentile(centres['x'], p),
                                         np.percentile(centres['x'], 100 - p)))
                   & (centres['y'].between(np.percentile(centres['y'], p),
                                           np.percentile(centres['y'], 100 - p)))].index
    core = np.isin(labels, kept)
    xlim = (x[core].min(), x[core].max())
    ylim = (y[core].min(), y[core].max())
    px = 0.03 * (xlim[1] - xlim[0]) or 0.5
    py = 0.03 * (ylim[1] - ylim[0]) or 0.5
    xlim, ylim = (xlim[0] - px, xlim[1] + px), (ylim[0] - py, ylim[1] + py)
    inside = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    fam_view = sorted(set(labels[inside & ~noise]))

    fig, ax = plt.subplots(figsize=(11, 7.6))

    ax.scatter(x[noise & inside], y[noise & inside], s=4, c='#e0e0e0', linewidths=0,
               label='noise')
    for fam in fam_view:
        m = (labels == fam) & inside
        ax.scatter(x[m], y[m], s=9, color=colors[fam], linewidths=0,
                   label=f'F{fam} (n={sizes[fam]})')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('umap1', fontsize=12)
    ax.set_ylabel('umap2', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_title(f'{len(fam_view)} of {len(families)} structural families in view\n'
                 f'({inside.sum()} of {len(x)} compounds)', fontsize=13)
    # Le famiglie sono oltre cento: la legenda non serve a identificarle una per una,
    # serve a dare le dimensioni. A corpo 5 era illeggibile in stampa (osservazione dei
    # coautori), quindi si sale a 7.5 e si passa a quattro colonne piu' strette.
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=4, fontsize=7.5,
              frameon=False, markerscale=2, handletextpad=0.3, columnspacing=0.8,
              labelspacing=0.3)

    fig.tight_layout()
    path = os.path.join(_figures_dir(out_dir), 'umap_clusters.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[figures] salvata {path}")
    return path


# Molecole discusse nel manoscritto, identificate per InChIKey e non per cluster: gli id di
# cluster cambiano da una run all'altra (la stessa molecola risulta in famiglie diverse su
# macchine diverse), mentre l'InChIKey e' stabile ed e' l'identificativo che il paper registra.
MANUSCRIPT_COMPOUNDS = (
    'XHCADAYNFIFUHF-TVKJYDDYSA-N',  # esculin (aesculin)
    'GRBHNQFQFHLCHO-BHMAJAPKSA-N',  # lupenone
    'ZCOLJUOHXJRHDI-CMWLGVBASA-N',  # genistin
)


def _read_optional_csv(path, what):
    """Legge un CSV accessorio segnalando l'assenza invece di sollevare, cosi' una figura
    resta producibile anche quando un artefatto di supporto manca."""
    if not path or not os.path.exists(path):
        print(f"[figures] {what} non disponibile: manca {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[figures] {what} non leggibile ({path}): {exc}")
        return None


def _first_synonym(value):
    """Primo nome dell'elenco di sinonimi separati da ';' del dizionario dei composti."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == 'nan':
        return None
    return text.split(';')[0].strip() or None


def compound_lookup(out_dir, centroids_inchikey_path=None, dictionary_path=None,
                    predictions_path=None):
    """Corrispondenze fra InChIKey, identificativo di molecola, famiglia, nome e probabilita'.

    Le probabilita' sono lette dagli output gia' salvati dalla pipeline, non ricalcolate, cosi'
    il valore mostrato coincide con quello della run che ha prodotto gli altri risultati.
    Restituisce sempre le due mappe: le voci non risolubili hanno campi a None.
    """
    centroids_inchikey_path = centroids_inchikey_path or f'{out_dir}/centroids_with_inchikey.csv'
    dictionary_path = dictionary_path or f'{out_dir}/inchikey_dict.csv'
    predictions_path = predictions_path or f'{out_dir}/test_data/TCMBANK/predictions_raw.csv'

    records = {}

    centroids = _read_optional_csv(centroids_inchikey_path, 'corrispondenza molecole/InChIKey')
    if centroids is not None:
        for row in centroids.itertuples(index=False):
            key = str(getattr(row, 'INCHIKEY', '')).strip()
            if not key:
                continue
            records[key] = {
                'inchikey': key,
                'mol_id': int(getattr(row, 'id_molecola')),
                'cluster': int(getattr(row, 'id_cluster')),
                'name': None,
                'smiles': None,
                'probability': None,
            }

    dictionary = _read_optional_csv(dictionary_path, 'dizionario dei composti')
    if dictionary is not None:
        for row in dictionary.itertuples(index=False):
            record = records.get(str(getattr(row, 'INCHIKEY', '')).strip())
            if record is None:
                continue
            record['name'] = _first_synonym(getattr(row, 'COMPOUNDS', None))
            smiles = getattr(row, 'SMILES', None)
            record['smiles'] = None if smiles is None else str(smiles)

    predictions = _read_optional_csv(predictions_path, 'probabilita' + "' predette")
    if predictions is not None and {'smiles', 'probability'}.issubset(predictions.columns):
        by_smiles = dict(zip(predictions['smiles'].astype(str), predictions['probability']))
        for record in records.values():
            if record['smiles'] in by_smiles:
                record['probability'] = float(by_smiles[record['smiles']])

    return {
        'by_inchikey': records,
        'by_mol_id': {record['mol_id']: record for record in records.values()},
    }


def _panel_title(record, mol_id, cluster=None, size=None):
    """Etichetta leggibile del pannello: ricade sull'identificativo di molecola quando il nome
    non e' noto e omette la probabilita' quando non e' reperibile."""
    record = record or {}
    parts = [record.get('name') or f'Molecule {mol_id}']
    if cluster is not None:
        parts.append(f'family {cluster}' + ('' if size is None else f' (n={size})'))
    probability = record.get('probability')
    if probability is not None:
        parts.append(f'p = {probability:.3f}')
    return ' - '.join(parts)


# Larghezza della figura in pollici: tenuta vicina a quella di una colonna di testo, cosi' che
# l'inclusione a \linewidth non riscali il risultato e le etichette restino della dimensione
# prevista invece di rimpicciolirsi.
FIGURE_WIDTH_IN = 7.0
# Spazio verticale riservato al titolo di ciascun pannello.
TITLE_SPACE_IN = 0.38
# Spazio verticale riservato alla legenda condivisa in fondo alla figura.
LEGEND_SPACE_IN = 0.75


# Legenda dell'attenzione, identica in ogni immagine per-molecola perche' le soglie sono fisse
# (vedi model_architecture/highlight_mol.py). Ripeterla in ogni pannello occupa spazio senza
# aggiungere informazione, quindi viene ritagliata dai pannelli e disegnata una volta sola.
ATTENTION_LEGEND = (
    ((1.0, 1.0, 0.0), '0.50 - 0.60'),
    ((1.0, 0.65, 0.0), '0.61 - 0.80'),
    ((1.0, 0.0, 0.0), '0.81 - 1.00'),
)


def _content_mask(image, tolerance=0.02):
    """Pixel che differiscono dal fondo, dedotto dall'angolo in alto a sinistra.

    Il canale alpha viene usato solo quando contiene davvero della trasparenza: queste immagini
    sono RGBA ma con fondo bianco opaco, quindi un test sull'alpha marcherebbe come contenuto
    l'intera tela e il ritaglio non farebbe nulla.
    """
    if not np.issubdtype(image.dtype, np.floating):
        tolerance = tolerance * 255

    if image.ndim == 2:
        return np.abs(image.astype(float) - float(image[0, 0])) > tolerance

    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        opaque = alpha.max()
        if (alpha < opaque - tolerance).any():
            return alpha > tolerance

    channels = image[:, :, :3].astype(float)
    return np.abs(channels - channels[0, 0]).max(axis=2) > tolerance


def _trim_uniform_border(image, tolerance=0.02, padding=8, drop_side_legend=True):
    """Ritaglia il fondo vuoto attorno alla molecola.

    Le molecole sono renderizzate su una tela molto piu' grande della struttura disegnata:
    senza ritaglio la maggior parte del pannello e' vuota e la molecola risulta minuscola.

    Con `drop_side_legend` viene scartata anche la legenda dei colori, che nelle immagini
    per-molecola e' ancorata fuori dagli assi (a destra, in alto) ed e' quindi separata dalla
    struttura da una banda verticale di fondo. I gruppi di colonne con contenuto vengono
    isolati e si scartano quelli piccoli che iniziano nella meta' destra: la molecola, che
    concentra la gran parte dei pixel, viene sempre conservata.
    """
    mask = _content_mask(image, tolerance)
    if not mask.any():
        return image

    columns = mask.sum(axis=0)
    if drop_side_legend:
        filled = columns > 0
        groups = []
        start = None
        for index, is_filled in enumerate(filled):
            if is_filled and start is None:
                start = index
            elif not is_filled and start is not None:
                groups.append((start, index))
                start = None
        if start is not None:
            groups.append((start, len(filled)))

        if len(groups) > 1:
            # La legenda e' ancorata a destra degli assi, quindi cade sempre dopo il gruppo che
            # contiene la struttura ed e' piu' stretta di quello. Si scarta cio' che segue il
            # gruppo piu' largo ed e' meno largo di esso: la molecola resta comunque intera,
            # comprese le etichette degli atomi adiacenti.
            widths = [b - a for a, b in groups]
            widest = int(np.argmax(widths))
            kept = [
                group for index, group in enumerate(groups)
                if index <= widest or (group[1] - group[0]) >= widths[widest]
            ]
            keep_mask = np.zeros_like(filled)
            for a, b in kept:
                keep_mask[a:b] = True
            mask = mask & keep_mask[None, :]

    if not mask.any():
        return image

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top = max(int(rows[0]) - padding, 0)
    bottom = min(int(rows[-1]) + 1 + padding, image.shape[0])
    left = max(int(cols[0]) - padding, 0)
    right = min(int(cols[-1]) + 1 + padding, image.shape[1])
    return image[top:bottom, left:right]


def _panel_axes(fig, count):
    """Assi dei pannelli.

    Con tre molecole la griglia e' una sopra centrata e due sotto: su una sola riga i pannelli
    sarebbero troppo stretti per una figura a colonna singola, e una griglia 2x2 lascerebbe un
    riquadro vuoto.
    """
    if count == 3:
        grid = fig.add_gridspec(2, 4)
        return [
            fig.add_subplot(grid[0, 1:3]),
            fig.add_subplot(grid[1, 0:2]),
            fig.add_subplot(grid[1, 2:4]),
        ]

    cols = count if count <= 2 else 2
    rows = int(np.ceil(count / cols))
    grid = fig.add_gridspec(rows, cols)
    return [fig.add_subplot(grid[index // cols, index % cols]) for index in range(count)]


def attention_maps(out_dir, centroids_path, labels_path, images_dir, n_panels=4,
                   inchikeys=None, centroids_inchikey_path=None, dictionary_path=None,
                   predictions_path=None):
    """Pannello di attention map gia' renderizzate durante la fase di predizione.

    Con `inchikeys` le molecole raffigurate sono quelle indicate, nell'ordine dato; senza, la
    selezione ricade sui medoidi delle famiglie piu' numerose. Ogni pannello riporta nome del
    composto, famiglia e probabilita' predetta quando reperibili.
    """
    if not os.path.isdir(images_dir):
        print(f"[figures] salto attention_maps: manca {images_dir}")
        return None

    lookup = compound_lookup(out_dir, centroids_inchikey_path, dictionary_path,
                             predictions_path)

    chosen = []
    if inchikeys:
        for key in inchikeys:
            record = lookup['by_inchikey'].get(key)
            if record is None:
                print(f"[figures] attention_maps: InChIKey non risolto, salto {key}")
                continue
            img_path = os.path.join(images_dir, f"mol_{record['mol_id']}_final.png")
            if not os.path.exists(img_path):
                print(f"[figures] attention_maps: manca l'immagine di {key} ({img_path}), salto")
                continue
            chosen.append((_panel_title(record, record['mol_id'], record['cluster']),
                           img_path, record['mol_id']))
    else:
        for p in (centroids_path, labels_path):
            if not os.path.exists(p):
                print(f"[figures] salto attention_maps: manca {p}")
                return None
        centroids = pd.read_csv(centroids_path)
        labels = pd.read_csv(labels_path).iloc[:, 0].values
        sizes = pd.Series(labels[labels >= 0]).value_counts()
        for cluster_id in sizes.index:
            row = centroids[centroids['id_cluster'] == cluster_id]
            if row.empty:
                continue
            mol_id = int(row['id_molecola'].iloc[0])
            img_path = os.path.join(images_dir, f'mol_{mol_id}_final.png')
            if os.path.exists(img_path):
                chosen.append((_panel_title(lookup['by_mol_id'].get(mol_id), mol_id,
                                            int(cluster_id), int(sizes[cluster_id])),
                               img_path, mol_id))
            if len(chosen) == n_panels:
                break

    if not chosen:
        print("[figures] salto attention_maps: nessuna immagine di medoide trovata")
        return None

    images = [_trim_uniform_border(mpimg.imread(img_path)) for _, img_path, _ in chosen]

    # Altezza derivata dalla forma effettiva delle molecole ritagliate, cosi' i pannelli non
    # restano ne' schiacciati ne' circondati da fondo vuoto.
    grid_cols = 2 if len(chosen) >= 3 else len(chosen)
    grid_rows = int(np.ceil(len(chosen) / grid_cols))
    panel_width = FIGURE_WIDTH_IN / grid_cols
    aspect = float(np.median([image.shape[1] / image.shape[0] for image in images]))
    fig_height = grid_rows * (panel_width / aspect + TITLE_SPACE_IN) + LEGEND_SPACE_IN

    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, fig_height))
    axes = _panel_axes(fig, len(chosen))

    for ax, image, (title, _img_path, _mol_id) in zip(axes, images, chosen):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    # Una sola legenda per l'intera figura: la scala e' la stessa in tutti i pannelli, quindi
    # i colori restano confrontabili fra molecole.
    fig.legend(
        handles=[Patch(facecolor=color, edgecolor='none', label=label)
                 for color, label in ATTENTION_LEGEND],
        loc='lower center', ncol=len(ATTENTION_LEGEND), frameon=False, fontsize=9,
        title='normalised attention', title_fontsize=9,
    )

    fig.tight_layout(rect=(0, LEGEND_SPACE_IN / fig_height, 1, 1), h_pad=1.6)
    path = os.path.join(_figures_dir(out_dir), 'attention_maps.pdf')
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"[figures] salvata {path}  (medoidi: {[c[2] for c in chosen]})")
    return path


def edgeshap_molecules(out_dir, scores_path, compounds_path):
    """Figura: l'albero delle clique di due composti, con gli archi colorati dal valore EdgeSHAPer.

    Il disegno vive in src/explain/, insieme al passo di attribuzione che produce i punteggi;
    qui si chiama soltanto perche' `generate_all` raccoglie in un unico comando tutte le figure
    del manoscritto. `compounds_path` serve solo a risalire dallo SMILES all'indice di molecola
    usato nel file dei punteggi, e va quindi letto nell'ORDINE DI INPUT: il file delle
    predizioni non va bene, perche' e' ordinato per SMILES.
    """
    for needed in (scores_path, compounds_path):
        if not os.path.exists(needed):
            print(f"[figures] salto edgeshap_molecules: manca {needed}")
            return None

    explain_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'explain')
    if explain_dir not in sys.path:
        sys.path.insert(0, explain_dir)
    import plot_edgeshaper_molecules

    path = os.path.join(_figures_dir(out_dir), 'edgeshap_molecules.pdf')
    plot_edgeshaper_molecules.main(['--scores', scores_path,
                                    '--compounds', compounds_path,
                                    '--out', path])
    return path


def generate_all(out_dir='../results', reference_predictions_path=None,
                 compounds_path='../data/tm-mc2/preprocessed_TMMC_compounds.csv'):
    """Genera tutte le figure a partire dagli artefatti standard della pipeline."""
    print("\nGenerating manuscript figures")
    print("################################# \n")

    tcm_score_distribution(
        out_dir,
        tcm_predictions_path=f'{out_dir}/test_data/TCMBANK/predictions_raw.csv',
        reference_predictions_path=reference_predictions_path,
    )
    umap_clusters(
        out_dir,
        embedding_path=f'{out_dir}/final_clustering/final_UMAP_embedding.csv',
        labels_path=f'{out_dir}/final_clustering/'
                    'final_UMAP_HDBSCAN_labels_min_samples_20_min_cluster_size_30.csv',
        centroids_path=f'{out_dir}/centroids_id_molecola.csv',
    )
    # Va per ultima: lo script che la disegna cambia lo stile globale di matplotlib (sfondo
    # bianco, palette propria), quindi messa prima contaminerebbe le altre figure.
    edgeshap_molecules(
        out_dir,
        scores_path=f'{out_dir}/explain/edgeshaper_edges.csv',
        compounds_path=compounds_path,
    )
