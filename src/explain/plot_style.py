"""Stile comune dei grafici: inchiostro tenue, griglia sottile, nessuna cornice.

Una sola definizione di colori e di stile per tutte le figure, cosi' i pannelli dei vari
script sono confrontabili fra loro. I valori vengono dalla palette validata per il
contrasto su sfondo chiaro.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# poli di una scala divergente: il segno del punteggio e' l'informazione, non la serie
NEGATIVE = "#e34948"
POSITIVE = "#2a78d6"

# ordine categorico validato; qui serve solo a separare gruppi adiacenti, non a dare identita'
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]


def apply_style():
    """Imposta i parametri globali di matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "font.size": 11,
        "figure.dpi": 150,
    })


def tidy(ax):
    """Griglia solo orizzontale, assi senza cornice."""
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
