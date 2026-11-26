#!/usr/bin/env python3
"""Figura: albero delle clique di una molecola, con gli archi colorati dal valore EdgeSHAPer.

La figura mostra l'unita' su cui lavora l'attribuzione. EdgeSHAPer assegna un valore di Shapley
a ogni ARCO dell'albero di giunzione, non ai legami del grafo atomo-legame: l'arco collega due
clique adiacenti e il suo valore misura quanto quella coppia contribuisce alla probabilita'
predetta. Disegnare il valore sui legami della molecola, come si fa con le mappe di attenzione,
nasconderebbe proprio questo. Qui la molecola resta sullo sfondo e sopra vi si sovrappone
l'albero: un nodo per clique, posto nel baricentro dei suoi atomi, e un arco per ogni coppia
adiacente, colorato dal suo valore di Shapley.

Gli indici delle clique vengono ricostruiti da zero con `DGLMolTree`, la stessa decomposizione
usata dalla pipeline, senza bisogno del modello. Che la ricostruzione coincida con quella da cui
provengono i punteggi viene verificato confrontando, arco per arco, lo SMILES del frammento
ottenuto da `unify_fragments` con quello registrato nel file dei punteggi.
"""
import argparse
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "gnn"))

import plot_style as style  # noqa: E402

CANVAS = "#ffffff"  # sfondo della figura: bianco puro, non il grigio chiaro dello stile

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Chem.Draw import rdMolDraw2D  # noqa: E402
from rdkit.Geometry import Point2D  # noqa: E402

from model_architecture.tree_building import DGLMolTree  # noqa: E402
from model_architecture.unify_high_mol import unify_fragments  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Due medoidi di famiglie chimicamente distinte, scelti fra i piu' compatti perche' la figura
# resti leggibile: un glucoside cumarinico (la classe ossano, il 40% delle occorrenze) e un
# sesquiterpene (i carbocicli saturi, il 29%).
DEFAULT_MOLECULES = [
    (16385, "Coumarin glucoside (family F41)"),
    (5492, "Sesquiterpene alcohol (family F97)"),
]


def tree_of(smiles):
    """Albero di giunzione della molecola, con gli indici di clique della pipeline."""
    tree = DGLMolTree(smiles)
    if tree.warning or not tree.nodes_dict:
        raise SystemExit(f"[fig] decomposizione fallita per {smiles}")
    return tree


def check_edges(tree, edges):
    """Confronta lo SMILES del frammento di ogni arco con quello registrato nei punteggi."""
    checked = agreeing = 0
    for row in edges.itertuples(index=False):
        recorded = getattr(row, "smiles", None)
        if not isinstance(recorded, str) or not recorded:
            continue
        submol, _ = unify_fragments(tree, [int(row.clique_a), int(row.clique_b)],
                                    deep_search=False)
        if submol is None:
            continue
        checked += 1
        try:
            rebuilt = Chem.MolToSmiles(submol)
        except Exception:
            continue
        reference = Chem.CanonSmiles(recorded) if Chem.MolFromSmiles(recorded) else recorded
        agreeing += int(rebuilt == reference or Chem.CanonSmiles(rebuilt) == reference)
    return checked, agreeing


def draw_molecule(mol, size):
    """Immagine della molecola e funzione che porta le coordinate molecolari in pixel."""
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    options = drawer.drawOptions()
    options.bondLineWidth = 2
    options.addStereoAnnotation = False
    options.clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    image = plt.imread(io.BytesIO(drawer.GetDrawingText()), format="png")

    conformer = mol.GetConformer()

    def to_pixels(atom_indices):
        points = [conformer.GetAtomPosition(i) for i in atom_indices]
        centre = Point2D(float(np.mean([p.x for p in points])),
                         float(np.mean([p.y for p in points])))
        drawn = drawer.GetDrawCoords(centre)
        return drawn.x, drawn.y

    return image, to_pixels


def panel(ax, smiles, title, edges, norm, cmap, size):
    mol = Chem.MolFromSmiles(smiles)
    image, to_pixels = draw_molecule(mol, size)
    ax.imshow(image, interpolation="antialiased")

    tree = tree_of(smiles)
    checked, agreeing = check_edges(tree, edges)
    print(f"[fig] {title}: {agreeing}/{checked} frammenti coincidono con i punteggi registrati")
    if checked and agreeing < checked:
        raise SystemExit("[fig] indici di clique non allineati: figura non prodotta")

    cliques = {i: tree.nodes_dict[i]["clique"] for i in range(len(tree.nodes_dict))}
    centres = {i: to_pixels(atoms) for i, atoms in cliques.items() if atoms}

    # prima gli archi deboli, cosi' i piu' forti restano visibili sopra
    ordered = edges.reindex(edges["edge_shapley"].abs().sort_values().index)
    for row in ordered.itertuples(index=False):
        a, b = int(row.clique_a), int(row.clique_b)
        if a not in centres or b not in centres:
            continue
        (x1, y1), (x2, y2) = centres[a], centres[b]
        value = float(row.edge_shapley)
        ax.plot([x1, x2], [y1, y2], color=cmap(norm(value)), alpha=0.9,
                linewidth=1.2 + 3.2 * min(abs(value) / 0.05, 1.0),
                solid_capstyle="round", zorder=2 + abs(value))

    for x, y in centres.values():
        ax.plot(x, y, marker="o", markersize=5.0, markerfacecolor=CANVAS,
                markeredgecolor=style.INK_SECONDARY, markeredgewidth=1.0, zorder=40)

    ax.set_title(title, fontsize=11, color=style.INK, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return len(cliques), len(edges)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", default="../results/explain/edgeshaper_edges.csv")
    parser.add_argument("--compounds",
                        default="../results/explain/tmmc_predictions_input_order.csv")
    parser.add_argument("--out", default="../results/explain/figures/edgeshap_molecules.pdf")
    parser.add_argument("--threshold", type=float, default=0.005,
                        help="soglia di selezione dei frammenti, segnata sulla barra colore")
    parser.add_argument("--limit", type=float, default=0.05,
                        help="estremo della scala di colore")
    args = parser.parse_args(argv)

    style.apply_style()
    # La figura finisce su una pagina bianca: lo sfondo chiaro dello stile
    # (#fcfcfb) si vede come un riquadro grigio attorno alle molecole, quindi
    # qui lo sfondo e' bianco puro. Il grigio dello stile resta sulle altre figure.
    plt.rcParams.update({"figure.facecolor": CANVAS,
                         "axes.facecolor": CANVAS,
                         "savefig.facecolor": CANVAS})
    compounds = pd.read_csv(args.compounds)
    scores = pd.read_csv(args.scores)

    wanted = [mol_id for mol_id, _ in DEFAULT_MOLECULES]
    selected = scores[scores["mol_index"].isin(wanted)]

    # Scala divergente: blu agli archi che abbassano la probabilita' predetta,
    # rosso a quelli che la alzano, neutro vicino allo zero.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "edgeshap", [style.POSITIVE, "#efeee8", style.NEGATIVE])
    norm = TwoSlopeNorm(vmin=-args.limit, vcenter=0.0, vmax=args.limit)

    size = (470, 330)
    fig = plt.figure(figsize=(8.6, 4.25))
    grid = fig.add_gridspec(2, len(DEFAULT_MOLECULES), height_ratios=[1.0, 0.065],
                            hspace=0.26, wspace=0.03)
    axes = [fig.add_subplot(grid[0, i]) for i in range(len(DEFAULT_MOLECULES))]
    cax = fig.add_subplot(grid[1, :])
    for ax, (mol_id, label) in zip(axes, DEFAULT_MOLECULES):
        edges = selected[selected["mol_index"] == mol_id]
        if edges.empty:
            raise SystemExit(f"[fig] nessun punteggio per la molecola {mol_id}")
        smiles = compounds["smiles"].iloc[mol_id]
        n_cliques, n_edges = panel(ax, smiles, label, edges, norm, cmap, size)
        print(f"[fig]   {n_cliques} clique, {n_edges} archi, "
              f"Shapley da {edges['edge_shapley'].min():+.4f} a "
              f"{edges['edge_shapley'].max():+.4f}")

    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                       orientation="horizontal", extend="both")
    bar.set_label(f"EdgeSHAPer value of the tree edge  "
                  f"(dashed: selection threshold $\\pm${args.threshold:g})",
                  fontsize=10.5, color=style.INK_SECONDARY, labelpad=5)
    bar.ax.tick_params(labelsize=9.5, colors=style.INK_SECONDARY, length=3)
    bar.outline.set_visible(False)
    for sign in (-1, 1):
        bar.ax.axvline(sign * args.threshold, color=style.INK, linewidth=0.9, linestyle="--")

    legend = [Line2D([], [], color=style.INK_SECONDARY, marker="o", linestyle="none",
                     markerfacecolor=CANVAS, markersize=4.5,
                     label="clique (junction-tree node)"),
              Line2D([], [], color=style.INK_MUTED, linewidth=3,
                     label="tree edge, coloured by its EdgeSHAPer value")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               frameon=False, fontsize=10, handletextpad=0.6, columnspacing=2.0)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.25)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", facecolor=CANVAS)
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200, bbox_inches="tight",
                facecolor=CANVAS)
    plt.close(fig)
    print(f"[fig] salvata {args.out}")


if __name__ == "__main__":
    main()
